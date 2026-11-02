"""The robot's abilities as plain methods, for the voice agent to call as tools.

One Desk owns the arm connection, the camera mapping, the last thing it saw, and the
places it has been told to remember. All coordinates it exchanges are desk centimetres:
the printed sheet's bottom-left corner is (0, 0), x runs along the sheet's bottom edge,
y up the sheet's left edge.
"""

import json
import threading
from pathlib import Path

import cv2
import numpy as np

from .arm import Arm, interruptible
from .camera import DeskMap, grab_frame
from .emotions import EMOTIONS
from .er2 import DetectorUnavailable, find_objects_er2
from .grasp import CLOSED, HELD_MIN, pick, place, plan_grasp, plan_place
from .kinematics import ArmModel
from .perception import DeskObject, _work_area_test, annotate, find_objects, load_background
from .poses import require_pose

ROOT = Path(__file__).resolve().parent.parent
PLACES_FILE = ROOT / "places.json"
VIEW_FILE = ROOT / "snapshots" / "agent_view.jpg"
VIEW_WIDTH = 1280
CLEARANCE_CM = 2.0   # keep a dropped object at least this far from other objects' outlines


class Desk:
    def __init__(self):
        self.arm = Arm().connect()
        self.desk = DeskMap.load()
        self.model = ArmModel.load()
        self.background = load_background()
        self.on_desk = _work_area_test()
        self.objects: list[DeskObject] = []
        self.lock = threading.Lock()   # one physical action at a time

    # ---- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.arm.shutdown()

    def stop(self) -> None:
        """Freeze the arm now (callable from any thread)."""
        self.arm.stop_event.set()

    def _park(self) -> None:
        self.arm.move_to(require_pose("neutral"), speed=45)
        self.arm.move_to(require_pose("rest"), speed=40)

    # ---- places --------------------------------------------------------------

    def places(self) -> dict[str, list[float]]:
        return json.loads(PLACES_FILE.read_text()) if PLACES_FILE.exists() else {}

    def remember_place(self, name: str, x_cm: float, y_cm: float) -> str:
        if not self.on_desk(x_cm / 100, y_cm / 100):
            return f"({x_cm:.1f}, {y_cm:.1f}) cm isn't on the usable desk area."
        places = self.places()
        places[name.strip().lower()] = [round(x_cm, 1), round(y_cm, 1)]
        PLACES_FILE.write_text(json.dumps(places, indent=1) + "\n")
        return f"Remembered '{name}' at ({x_cm:.1f}, {y_cm:.1f}) cm."

    def forget_place(self, name: str) -> str:
        places = self.places()
        if places.pop(name.strip().lower(), None) is None:
            return f"No place called '{name}'."
        PLACES_FILE.write_text(json.dumps(places, indent=1) + "\n")
        return f"Forgot '{name}'."

    # ---- looking -------------------------------------------------------------

    def look(self) -> tuple[bytes, dict]:
        """Park out of the way, photograph the desk, find objects.
        -> (JPEG of the annotated view, summary for the agent)."""
        with self.lock:
            self._park()
            stop = self.arm.stop_event
            frame = interruptible(stop, grab_frame)
            try:
                self.objects, info = interruptible(stop, find_objects_er2, frame, self.background, self.desk,
                                                   self.model, self.arm.limits)
                info["detector"] = "gemini-robotics-er-2"
            except DetectorUnavailable as e:
                # Fall back to the empty-desk comparison: positions and grips are still exact,
                # but there are no names, and touching objects may merge into one outline.
                self.objects = find_objects(frame, self.background, self.desk, self.model, self.arm.limits)
                info = {"seconds": 0.0, "detector": "fallback",
                        "note": f"{e}. Using the basic detector: objects have no labels (identify them "
                                "from the photo by their numbers), and touching objects may be merged "
                                "into one outline."}
            self.last_frame = frame
        view = self._draw_view(frame)
        VIEW_FILE.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(VIEW_FILE), view)
        small = cv2.resize(view, (VIEW_WIDTH, round(view.shape[0] * VIEW_WIDTH / view.shape[1])))
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
        summary = {
            "objects": [{**o.summary(), "length_mm": round(o.length_mm), "width_mm": round(o.width_mm),
                         "why_not": None if o.graspable else ("out of reach" if not o.reachable else "too wide for the gripper")}
                        for o in self.objects],
            "places": self.places(),
            "detector_seconds": round(info["seconds"], 1),
            "detector": info["detector"],
        }
        if info.get("note"):
            summary["detector_note"] = info["note"]
        for s in summary["objects"]:
            s.pop("size_mm", None)
        return jpg, summary

    def _draw_view(self, frame: np.ndarray) -> np.ndarray:
        """Objects plus a labelled 5 cm desk grid and remembered places, so coordinates can
        be read straight off the picture."""
        out = frame.copy()
        h, w = out.shape[:2]
        for v in range(-30, 55, 5):
            for line, lab in ((np.c_[np.full(100, v), np.linspace(-30, 55, 100)], f"x={v}"),
                              (np.c_[np.linspace(-30, 55, 100), np.full(100, v)], f"y={v}")):
                pts = self.desk.to_pixel(line / 100)
                ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                ok &= np.array([self.on_desk(*(p / 100)) for p in line])
                if ok.sum() < 2:
                    continue
                seg = pts[ok].astype(np.int32)
                color = (0, 140, 255) if v % 10 == 0 else (0, 200, 255)
                cv2.polylines(out, [seg], False, color, 1, cv2.LINE_AA)
                if v % 10 == 0:
                    tx, ty = seg[len(seg) // 2]
                    cv2.putText(out, lab, (int(tx) + 3, int(ty) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 90, 200), 2)
        out = annotate(out, self.objects, self.desk)
        for name, (x, y) in self.places().items():
            p = tuple(int(v) for v in self.desk.to_pixel((x / 100, y / 100)))
            cv2.drawMarker(out, p, (255, 0, 0), cv2.MARKER_DIAMOND, 26, 3)
            cv2.putText(out, name, (p[0] + 14, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        return out

    # ---- acting --------------------------------------------------------------

    def move_object(self, object_id: int, x_cm: float | None = None, y_cm: float | None = None,
                    place_name: str | None = None, next_to_id: int | None = None) -> str:
        """Pick up object #object_id (from the latest look) and put it at a spot, a named place,
        or beside another object. The destination is fully checked before the arm moves."""
        obj = next((o for o in self.objects if o.id == object_id), None)
        if obj is None:
            return f"There's no object #{object_id} in the latest look. Look again first."
        if not obj.graspable:
            return f"#{object_id} ({obj.label}) can't be picked up: " + (
                "it's out of reach." if not obj.reachable else f"it's {obj.width_mm:.0f} mm wide, too wide for the gripper.")
        if self._grasp(obj) is None:
            return f"#{object_id} ({obj.label}) can't be picked up: I can't line the gripper up with it."
        if next_to_id is not None:
            other = next((o for o in self.objects if o.id == next_to_id), None)
            if other is None or other is obj:
                return f"There's no other object #{next_to_id} in the latest look. Look again first."
            spot = self.spot_next_to(obj, other)
            if spot is None:
                return (f"There's no free spot within reach beside #{next_to_id} ({other.label}) for "
                        f"#{object_id}. Moving #{next_to_id} somewhere roomier first might help.")
            x_cm, y_cm = spot
        elif place_name:
            spot = self.places().get(place_name.strip().lower())
            if spot is None:
                return f"No remembered place called '{place_name}'. Known: {', '.join(self.places()) or 'none'}."
            x_cm, y_cm = spot
        if x_cm is None or y_cm is None:
            return "Give a destination: x_cm and y_cm, a place_name, or next_to_id."
        why = self.check_destination(obj, x_cm, y_cm)
        if why:
            return f"Didn't move #{object_id} ({obj.label}): {why} Nothing was touched."
        with self.lock, self.arm.busy():   # one motion for the music, camera check included
            err = pick(self.arm, self.model, obj.x, obj.y, obj.grasp_deg, obj.width_mm)
            if err is None:
                # Second opinion from the camera: carry it over the destination (out of the
                # way of its old spot) and check that the spot is now empty.
                self._carry_above(x_cm / 100, y_cm / 100, obj.width_mm)
                seen = self._spot_check(obj)
                self.last_check = seen
                if seen == "still there":
                    err = ("the gripper felt something, but the camera still sees it at its old spot, "
                           "so the grab missed")
            if err is None:
                err = place(self.arm, self.model, x_cm / 100, y_cm / 100, held_width_mm=obj.width_mm)
            if err and self.arm.read()["gripper"] >= HELD_MIN:
                place(self.arm, self.model, obj.x, obj.y, held_width_mm=obj.width_mm)
                err += " (I put it back where it was.)"
        # The moved object's outline is stale now (and a failed attempt may have nudged it):
        # drop it until the next look. The others haven't moved, so they stay usable.
        self.objects = [o for o in self.objects if o is not obj]
        if err:
            return f"Couldn't move #{object_id} ({obj.label}): {err}"
        return (f"Moved #{object_id} ({obj.label}) to ({x_cm:.1f}, {y_cm:.1f}) cm. The other objects' "
                "ids still work; look again before moving this one again or to check the result.")

    # ---- destination planning (no arm motion) ---------------------------------

    def _grasp(self, obj: DeskObject):
        """The grab plan for obj (cached: IK is slow-ish and the destination search asks often)."""
        key = (id(obj), obj.x, obj.y, obj.grasp_deg)
        if getattr(self, "_grasp_key", None) != key:
            self._grasp_key = key
            self._grasp_plan = plan_grasp(self.model, self.arm.limits, obj.x, obj.y, obj.grasp_deg,
                                          obj.width_mm, seed=self.arm.read())
        return self._grasp_plan

    def _crowding(self, obj: DeskObject, x_cm: float, y_cm: float) -> DeskObject | None:
        """The first other object whose outline is too close to (x_cm, y_cm) for obj, if any."""
        p = tuple(float(v) for v in self.desk.to_pixel((x_cm / 100, y_cm / 100)))
        for o in self.objects:
            if o is obj:
                continue
            gap_cm = -cv2.pointPolygonTest(o.contour_px.astype(np.float32), p, True) * self._cm_per_px(o)
            if gap_cm < CLEARANCE_CM + obj.width_mm / 20:   # distance outside its outline
                return o
        return None

    def _reachable(self, obj: DeskObject, x_cm: float, y_cm: float) -> bool:
        """Plan the whole move (grab obj, carry, set down at the spot) without moving."""
        g = self._grasp(obj)
        if g is None:
            return False
        high = g[2]   # the pose we'd be in, holding it, when the carry starts
        return plan_place(self.model, self.arm.limits, x_cm / 100, y_cm / 100, high["wrist_roll"], high,
                          obj.width_mm) is not None

    def check_destination(self, obj: DeskObject, x_cm: float, y_cm: float) -> str | None:
        """-> why obj can't go to (x_cm, y_cm), or None if it can."""
        if not self.on_desk(x_cm / 100, y_cm / 100):
            return f"({x_cm:.1f}, {y_cm:.1f}) cm isn't on the usable desk area."
        o = self._crowding(obj, x_cm, y_cm)
        if o is not None:
            return f"({x_cm:.1f}, {y_cm:.1f}) cm is on or too close to #{o.id} ({o.label})."
        if not self._reachable(obj, x_cm, y_cm):
            return f"({x_cm:.1f}, {y_cm:.1f}) cm is out of the arm's reach for this object."
        return None

    def spot_next_to(self, obj: DeskObject, other: DeskObject) -> tuple[float, float] | None:
        """The closest free, reachable spot for obj beside other: alongside it (parallel, a
        couple of cm apart) first, sliding along its length if needed, then off its ends."""
        u = np.array([np.cos(np.radians(other.axis_deg)), np.sin(np.radians(other.axis_deg))])  # long side
        n = np.array([-u[1], u[0]])
        c = np.array([other.x, other.y]) * 100                       # cm
        cands = []
        for gap in (CLEARANCE_CM + 0.5, 4.0, 6.0):
            side = other.width_mm / 20 + obj.width_mm / 20 + gap
            for shift in (0.0, 3.0, -3.0, 6.0, -6.0):
                for s in (1, -1):
                    cands.append((gap + abs(shift) / 2, c + s * side * n + shift * u))
            end = other.length_mm / 20 + obj.width_mm / 20 + gap
            for s in (1, -1):
                cands.append((gap + 4.0, c + s * end * u))
        cands.sort(key=lambda t: t[0])
        for _, (x, y) in cands:
            if self.on_desk(x / 100, y / 100) and self._crowding(obj, x, y) is None \
                    and self._reachable(obj, x, y):
                return round(float(x), 1), round(float(y), 1)
        return None

    def _carry_above(self, x: float, y: float, held_width_mm: float = 0.0) -> None:
        seed = self.arm.read()
        pts = plan_place(self.model, self.arm.limits, x, y, seed["wrist_roll"], seed, held_width_mm)
        if pts is not None:
            self.arm.move_to({**pts[0], "gripper": CLOSED}, speed=40)

    def _spot_check(self, obj: DeskObject) -> str:
        """Compare the object's old spot now vs. at the last look and vs. the empty desk.
        -> "gone", "still there", or "unclear" (the arm blocks the view, or the object
        barely differs from the empty desk, e.g. white on the white sheet)."""
        before = getattr(self, "last_frame", None)
        if before is None:
            return "unclear"
        now = interruptible(self.arm.stop_event, grab_frame)
        region = np.zeros(now.shape[:2], np.uint8)
        cv2.drawContours(region, [obj.contour_px], -1, 255, -1)
        region = cv2.erode(region, np.ones((9, 9), np.uint8)) > 0
        if region.sum() < 200:
            return "unclear"
        lab = lambda im: cv2.cvtColor(cv2.GaussianBlur(im, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        a, b, bg = lab(now), lab(before), lab(self.background)
        near = lambda x, y: (np.sqrt(((x - y) ** 2).sum(axis=2)) < 14)[region].mean()
        if near(b, bg) > 0.5:
            return "unclear"            # the object hardly showed against the empty desk
        if near(a, bg) > 0.6:
            return "gone"
        if near(a, b) > 0.6:
            return "still there"
        return "unclear"                # likely the arm's own shadow or body over the spot

    def _cm_per_px(self, o: DeskObject) -> float:
        a = self.desk.to_world(np.array([[0.0, 0.0], [100.0, 0.0]]) + self.desk.to_pixel((o.x, o.y)))
        return float(np.linalg.norm(a[1] - a[0]) * 100 / 100)

    def emote(self, emotion: str, on_cue=None) -> str:
        """on_cue() is called the instant the expressive motion starts (for a matching sound)."""
        fn = EMOTIONS.get(emotion)
        if fn is None:
            return f"Unknown emotion '{emotion}'. Options: {', '.join(EMOTIONS)}."
        with self.lock:
            fn(self.arm, on_cue=on_cue)
        return f"Showed {emotion}."

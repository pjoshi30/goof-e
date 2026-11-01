"""Find objects on the desk by comparing the camera image to a photo of the empty desk.

The camera is fixed and the arm parks at 'rest' before every look, so anything that
differs from the saved background is an object. For each one we report where it is on
the desk, how big it is, and which way to close the gripper across it. Naming objects
and telling touching ones apart is done by Gemini Robotics-ER 2 (see er2.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import DeskMap
from .kinematics import ArmModel
from .targets import OUT_DIR

BACKGROUND_FILE = OUT_DIR / "background.png"
WORK_AREA_FILE = OUT_DIR / "work_area.json"   # desk outline (world m); objects outside it are ignored

DIFF_THRESHOLD = 14.0      # colour distance (LAB units) that counts as "changed"
GROUP_PX = 15              # pieces this close (pixels, ~6 mm on the desk) count as one object
MIN_AREA_CM2 = 1.0         # smaller blobs are noise (crumbs, glare)
MAX_AREA_CM2 = 400.0       # bigger ones are lighting changes or the arm, not objects
GRASP_Z = 0.004            # height the jaws close at, m (same as grasp.GRASP_Z)
BASE_CLEAR_M = 0.08        # ignore everything this close to the base (the arm itself)
MAX_GRIP_MM = 68.0         # widest footprint the jaws can close around (camera sizes run a few mm big)


@dataclass
class DeskObject:
    id: int
    x: float                 # desk position of the centre, m
    y: float
    length_mm: float         # long side of the footprint
    width_mm: float          # short side: what the gripper closes across
    axis_deg: float          # direction of the long side on the desk, degrees (world frame)
    reach_m: float           # distance from the arm's base
    reachable: bool          # the arm has an IK solution to grasp it
    contour_px: np.ndarray   # outline in the image, for drawing
    label: str = ""          # name, when a vision model provided one ("blue marker")

    @property
    def graspable(self) -> bool:
        return self.reachable and self.width_mm <= MAX_GRIP_MM

    @property
    def grasp_deg(self) -> float:
        """Direction to close the jaws: across the short side."""
        return (self.axis_deg + 90.0) % 180.0

    def summary(self) -> dict:
        return {"id": self.id, **({"label": self.label} if self.label else {}), "x_cm": round(self.x * 100, 1), "y_cm": round(self.y * 100, 1),
                "size_mm": [round(self.length_mm), round(self.width_mm)],
                "reach_cm": round(self.reach_m * 100, 1), "graspable": self.graspable}


def save_background(img: np.ndarray, path: Path = BACKGROUND_FILE) -> None:
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), img)


def load_background(path: Path = BACKGROUND_FILE) -> np.ndarray:
    if not path.exists():
        raise SystemExit("No empty-desk photo yet. Clear the desk, park the arm, then run: "
                         "python -m deskbot.perceive --background")
    return cv2.imread(str(path))


def save_work_area(desk_px, ignore_px, desk: DeskMap, path: Path = WORK_AREA_FILE) -> None:
    """Store, as desk coordinates, the desk outline and the parked arm's footprint
    (both traced in image pixels). Objects are only looked for on the desk, off the arm."""
    w = lambda poly: np.round(desk.to_world(np.asarray(poly, float)), 4).tolist()
    path.write_text(json.dumps({"desk_m": w(desk_px), "ignore_m": [w(p) for p in ignore_px]}) + "\n")


def _work_area_test(path: Path = WORK_AREA_FILE):
    """-> function(x_m, y_m) saying whether a point is on the usable desk."""
    if not path.exists():
        return lambda x, y: True
    d = json.loads(path.read_text())
    poly = lambda pts: (np.array(pts) * 1000).astype(np.float32).reshape(-1, 1, 2)
    inside = lambda pts, x, y: cv2.pointPolygonTest(poly(pts), (x * 1000, y * 1000), False) >= 0
    return lambda x, y: inside(d["desk_m"], x, y) and not any(inside(p, x, y) for p in d["ignore_m"])


def change_mask(img: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Pixels that differ from the empty desk, cleaned up into solid blobs."""
    a = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = cv2.cvtColor(cv2.GaussianBlur(background, (7, 7), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = a - b
    d[..., 0] -= np.median(d[..., 0])   # overall brightness drift (clouds, lamp) isn't an object
    # Shadows only darken, and only a little: count darkening at half weight so a shadow
    # stays under the threshold while a dark object on white paper still stands out.
    d[..., 0] = np.where(d[..., 0] < 0, 0.5, 1.0) * d[..., 0]
    dist = np.sqrt((d ** 2).sum(axis=2))
    mask = (dist > DIFF_THRESHOLD).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


def find_objects(img: np.ndarray, background: np.ndarray, desk: DeskMap,
                 model: ArmModel, limits: dict) -> list[DeskObject]:
    mask = change_mask(img, background)
    # A white object on the white sheet often shows up as a few separate pieces (only its
    # shaded edges differ). Pieces closer than ~6 mm are one object: group them, then
    # outline each group by the convex hull of its pixels.
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (GROUP_PX, GROUP_PX)))
    n, labels = cv2.connectedComponents(near)
    contours = []
    for k in range(1, n):
        ys, xs = np.nonzero((labels == k) & (mask > 0))
        if len(xs):
            contours.append(cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)))
    on_desk = _work_area_test()
    img_h, img_w = mask.shape
    found = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bx <= 1 or by <= 1 or bx + bw >= img_w - 1 or by + bh >= img_h - 1:
            continue  # cut off by the frame edge: can't tell its true size or centre
        obj = describe(c, desk, model, limits, on_desk)
        if obj is not None:
            found.append(obj)
    found.sort(key=lambda o: o.reach_m)
    for i, o in enumerate(found, 1):
        o.id = i
    return found


def describe(c: np.ndarray, desk: DeskMap, model: ArmModel, limits: dict, on_desk=None,
             label: str = "", check_area: bool = True) -> DeskObject | None:
    """Outline in pixels -> DeskObject (desk position, footprint, grip angle, reachability),
    or None if it's noise, off the desk, or the arm itself."""
    on_desk = on_desk or _work_area_test()
    world_mm = desk.to_world(c.reshape(-1, 2).astype(float)) * 1000
    area_cm2 = cv2.contourArea(world_mm.astype(np.float32)) / 100
    if check_area and not MIN_AREA_CM2 <= area_cm2 <= MAX_AREA_CM2:
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect(world_mm.astype(np.float32))
    centre = np.array([cx, cy]) / 1000
    reach = float(np.linalg.norm(centre - np.asarray(model.base_xyz[:2], float)))
    if reach < BASE_CLEAR_M or not on_desk(*centre):
        return None  # the arm itself, the floor, or the wall
    # minAreaRect's angle belongs to the side of length w; make it the long side's.
    length, width, axis = (w, h, ang) if w >= h else (h, w, ang + 90.0)
    reachable = all(model.ik((centre[0], centre[1], z), limits) is not None for z in (GRASP_Z, 0.07))
    return DeskObject(0, float(centre[0]), float(centre[1]), float(length), float(width),
                      float(axis % 180.0), reach, reachable, c, label)


def annotate(img: np.ndarray, objects: list[DeskObject], desk: DeskMap) -> np.ndarray:
    """Numbered outlines plus the gripper closing direction, for people and for Claude."""
    out = img.copy()
    for o in objects:
        color = (0, 220, 0) if o.graspable else (0, 165, 255) if o.reachable else (160, 160, 160)
        cv2.drawContours(out, [o.contour_px], -1, color, 3)
        g = np.radians(o.grasp_deg)
        half = (o.width_mm / 2 + 12) / 1000
        ends = desk.to_pixel(np.array([[o.x - half * np.cos(g), o.y - half * np.sin(g)],
                                       [o.x + half * np.cos(g), o.y + half * np.sin(g)]]))
        cv2.line(out, tuple(ends[0].astype(int)), tuple(ends[1].astype(int)), (255, 0, 255), 3)
        cx, cy = desk.to_pixel((o.x, o.y)).astype(int)
        label = f"{o.id} {o.label}" if o.label else str(o.id)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
        cv2.rectangle(out, (cx + 18, cy - th - 22), (cx + 30 + tw, cy - 10), (0, 0, 0), -1)
        cv2.putText(out, label, (cx + 24, cy - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
    return out

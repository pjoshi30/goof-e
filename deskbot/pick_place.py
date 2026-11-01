"""Click an object, then click where it should go; the arm moves it.

    python -m deskbot.pick_place            # run from Terminal (needs the camera)
    python -m deskbot.pick_place --classic  # without Gemini: empty-desk comparison only

The arm parks at 'rest' before every photo so it doesn't hide anything.
Window keys: r = look again, q = quit (arm parks at 'rest').
"""

import sys

import cv2
import numpy as np

from .arm import Arm
from .camera import DeskMap, grab_frame
from .grasp import HELD_MIN, pick, place
from .kinematics import ArmModel
from .perception import annotate, find_objects, load_background
from .poses import require_pose

WINDOW = "pick & place (click object, then destination / r / q)"


def park(arm: Arm) -> None:
    arm.move_to(require_pose("neutral"), speed=45)
    arm.move_to(require_pose("rest"), speed=40)


def nearest_object(objects, xy, max_m: float = 0.04):
    best = min(objects, key=lambda o: np.hypot(o.x - xy[0], o.y - xy[1]), default=None)
    return best if best and np.hypot(best.x - xy[0], best.y - xy[1]) <= max_m else None


def main() -> None:
    desk, model, background = DeskMap.load(), ArmModel.load(), load_background()
    clicks: list[tuple[int, int]] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    with Arm() as arm:
        def look():
            park(arm)
            frame = grab_frame()
            if "--classic" in sys.argv:
                objs = find_objects(frame, background, desk, model, arm.limits)
            else:
                from .er2 import find_objects_er2
                objs, info = find_objects_er2(frame, background, desk, model, arm.limits)
                print(f"  (ER 2: {info['seconds']:.1f} s)")
            for o in objs:
                print(f"  #{o.id} {o.label}: ({o.x*100:.1f}, {o.y*100:.1f}) cm, {o.length_mm:.0f} x {o.width_mm:.0f} mm"
                      + ("" if o.graspable else "  (can't grab)"))
            return frame, objs

        frame, objects = look()
        view = annotate(frame, objects, desk)
        cv2.imshow(WINDOW, view)
        chosen = None
        print("Click an object.")
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frame, objects = look()
                view, chosen = annotate(frame, objects, desk), None
                cv2.imshow(WINDOW, view)
            if not clicks:
                continue
            px = clicks.pop()
            clicks.clear()
            xy = desk.to_world(px)
            if chosen is None:
                chosen = nearest_object(objects, xy)
                if chosen is None:
                    print("  no object there")
                elif not chosen.graspable:
                    why = "out of reach" if not chosen.reachable else f"too wide ({chosen.width_mm:.0f} mm)"
                    print(f"  #{chosen.id} can't be grabbed: {why}")
                    chosen = None
                else:
                    print(f"  #{chosen.id} selected. Click where it should go.")
                    marked = view.copy()
                    cv2.drawMarker(marked, tuple(int(v) for v in desk.to_pixel((chosen.x, chosen.y))),
                                   (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 4)
                    cv2.imshow(WINDOW, marked)
                continue
            dest = xy
            cv2.circle(view, tuple(px), 14, (0, 0, 255), 3)
            cv2.imshow(WINDOW, view)
            cv2.waitKey(1)
            print(f"  moving #{chosen.id} to ({dest[0]*100:.1f}, {dest[1]*100:.1f}) cm")
            err = pick(arm, model, chosen.x, chosen.y, chosen.grasp_deg, chosen.width_mm)
            if err is None:
                err = place(arm, model, *dest, held_width_mm=chosen.width_mm)
            print(f"  {err}" if err else "  done")
            if err and arm.read()["gripper"] >= HELD_MIN:
                print("  still holding it: putting it back down where it was")
                place(arm, model, chosen.x, chosen.y, held_width_mm=chosen.width_mm)
            frame, objects = look()
            view, chosen = annotate(frame, objects, desk), None
            cv2.imshow(WINDOW, view)
            print("Click an object.")
        park(arm)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Find objects on the desk.

    python -m deskbot.perceive --background   # save the empty desk (arm parked at rest)
    python -m deskbot.perceive                # list objects, write snapshots/objects.jpg
    python -m deskbot.perceive --er2          # same, but Gemini Robotics-ER 2 finds and names them

Both read snapshots/cam0.jpg (take one with snap.command); add --live to grab a fresh
frame instead (needs a Terminal with camera access).
"""

import sys
from pathlib import Path

import cv2

from .arm import joint_limits
from .camera import DeskMap, grab_frame
from .kinematics import ArmModel
from .perception import BACKGROUND_FILE, annotate, find_objects, load_background, save_background

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "snapshots" / "cam0.jpg"
OVERLAY = ROOT / "snapshots" / "objects.jpg"


def main() -> None:
    img = grab_frame() if "--live" in sys.argv else cv2.imread(str(SNAPSHOT))
    if img is None:
        raise SystemExit(f"No image at {SNAPSHOT}. Take one with snap.command first.")
    if "--background" in sys.argv:
        save_background(img)
        print(f"saved empty-desk photo to {BACKGROUND_FILE}")
        return

    desk, model = DeskMap.load(), ArmModel.load()
    if "--er2" in sys.argv:
        from .er2 import find_objects_er2
        objects, info = find_objects_er2(img, load_background(), desk, model, joint_limits())
        print(f"ER 2 answered in {info['seconds']:.1f} s with {len(info['er2'])} object(s)"
              + (f"; box only (no clean outline) for: {', '.join(info['box_fallback'])}" if info["box_fallback"] else ""))
    else:
        objects = find_objects(img, load_background(), desk, model, joint_limits())
    cv2.imwrite(str(OVERLAY), annotate(img, objects, desk))
    print(f"{len(objects)} object(s); overlay at {OVERLAY}")
    for o in objects:
        flag = "ok" if o.graspable else "too wide" if o.reachable else "out of reach"
        print(f"  #{o.id} {o.label or ''}: ({o.x*100:5.1f}, {o.y*100:5.1f}) cm  {o.length_mm:4.0f} x {o.width_mm:3.0f} mm"
              f"  grip at {o.grasp_deg:5.1f} deg  reach {o.reach_m*100:4.1f} cm  {flag}")


if __name__ == "__main__":
    main()

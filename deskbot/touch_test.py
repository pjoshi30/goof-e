"""Click a spot in the camera image; the arm touches it (stops 1 cm above the desk).

    python -m deskbot.touch_test            # live camera (run from Terminal)
    python -m deskbot.touch_test --image snapshots/cam0.jpg

In the window: click = touch that spot, r = refresh image, q = quit (arm parks at 'rest').
The touched spot is printed in desk coordinates; compare it with where the tip lands.
"""

import sys

import cv2

from .arm import Arm
from .camera import DeskMap, grab_frame
from .kinematics import ArmModel
from .motion import touch_point

WINDOW = "touch test (click / r / q)"


def main() -> None:
    desk = DeskMap.load()
    model = ArmModel.load()
    image_path = sys.argv[sys.argv.index("--image") + 1] if "--image" in sys.argv else None

    def get_frame():
        return cv2.imread(image_path) if image_path else grab_frame()

    def annotate(img, click=None):
        out = img.copy()
        base = tuple(int(v) for v in desk.to_pixel(model.base_xyz[:2]))
        cv2.drawMarker(out, base, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        cv2.putText(out, "arm base", (base[0] + 10, base[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if click:
            cv2.circle(out, click, 10, (0, 255, 0), 2)
        return out

    frame = get_frame()
    clicks: list[tuple[int, int]] = []
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.setMouseCallback(WINDOW, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    cv2.imshow(WINDOW, annotate(frame))

    with Arm() as arm:
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                frame = get_frame()
                cv2.imshow(WINDOW, annotate(frame))
            if clicks:
                px = clicks.pop()
                clicks.clear()
                x, y = desk.to_world(px)
                print(f"click {px} -> desk ({x*100:.1f}, {y*100:.1f}) cm")
                cv2.imshow(WINDOW, annotate(frame, px))
                cv2.waitKey(1)
                err = touch_point(arm, model, x, y)
                print(f"  {err}" if err else "  touched — how far off was it?")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

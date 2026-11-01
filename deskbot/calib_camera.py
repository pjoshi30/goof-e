"""Calibrate camera pixels -> desk coordinates from the printed sheet.

    python -m deskbot.calib_camera                # uses snapshots/cam0.jpg
    python -m deskbot.calib_camera --live         # grabs a fresh frame (needs camera access)

Writes calib/camera.json and calib/camera_check.jpg (5 cm grid + targets drawn on the image).
After this, the sheet can be removed; only the camera must not move.
"""

import sys
from pathlib import Path

import cv2

from .camera import DeskMap, detect_markers, draw_grid, grab_frame
from .targets import OUT_DIR, touch_targets

SNAPSHOT = Path(__file__).resolve().parent.parent / "snapshots" / "cam0.jpg"


def main() -> None:
    img = grab_frame() if "--live" in sys.argv else cv2.imread(str(SNAPSHOT))
    if img is None:
        raise SystemExit(f"Couldn't read {SNAPSHOT}. Take a snapshot first.")
    markers = detect_markers(img)
    print(f"markers found: {sorted(markers)}")
    desk, err_mm = DeskMap.from_markers(markers, img.shape[1::-1])
    print(f"max corner error: {err_mm:.2f} mm")
    if len(markers) < 4:
        print("warning: not all 4 markers visible — accuracy away from the sheet will suffer.")

    check = draw_grid(img, desk)
    for name, xy in touch_targets().items():
        p = tuple(int(v) for v in desk.to_pixel(xy))
        cv2.circle(check, p, 6, (255, 0, 255), 2)
        cv2.putText(check, name, (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    for c in markers.values():
        cv2.polylines(check, [c.astype(int)], True, (0, 255, 0), 2)

    desk.save()
    cv2.imwrite(str(OUT_DIR / "camera_check.jpg"), check)
    print(f"saved {OUT_DIR / 'camera.json'} and camera_check.jpg")


if __name__ == "__main__":
    main()

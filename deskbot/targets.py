"""The printed calibration sheet (US Letter) and the desk coordinate frame it defines.

    python -m deskbot.targets      # writes calib/targets.pdf — print at 100% / "Actual size"

World frame: meters, origin at the sheet's bottom-left corner, x to the right,
y toward the top of the page, z up out of the desk.

The sheet has 4 ArUco markers (ids 0-3) for the camera and 9 numbered crosshairs
the gripper tip touches during arm calibration. Marker centers are touch targets
too (m0..m3).
"""

from pathlib import Path

import cv2
import numpy as np

PAGE_W_CM, PAGE_H_CM = 21.59, 27.94
MARKER_CM = 4.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Page coordinates in cm, measured from the top-left of the page (y down).
_MARKER_CENTERS_PAGE = {0: (3.5, 3.5), 1: (18.09, 3.5), 2: (18.09, 24.44), 3: (3.5, 24.44)}
_CROSS_X = (5.8, 10.8, 15.8)
_CROSS_Y = (8.5, 14.0, 19.5)
_SCALE_BAR = ((6.0, 3.5), (16.0, 3.5))  # 10 cm, between the top markers

OUT_DIR = Path(__file__).resolve().parent.parent / "calib"


def page_to_world(x_cm: float, y_cm: float) -> tuple[float, float]:
    return x_cm / 100.0, (PAGE_H_CM - y_cm) / 100.0


def marker_corners_world(marker_id: int) -> np.ndarray:
    """4x2 corners in ArUco order (TL, TR, BR, BL as printed), world meters."""
    cx, cy = _MARKER_CENTERS_PAGE[marker_id]
    h = MARKER_CM / 2
    page = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
    return np.array([page_to_world(x, y) for x, y in page])


def touch_targets() -> dict[str, tuple[float, float]]:
    """Name -> (x, y) world meters for every point the gripper tip can touch."""
    t = {}
    n = 1
    for y in _CROSS_Y:
        for x in _CROSS_X:
            t[str(n)] = page_to_world(x, y)
            n += 1
    for mid, (x, y) in _MARKER_CENTERS_PAGE.items():
        t[f"m{mid}"] = page_to_world(x, y)
    return t


# ---- printable sheet ---------------------------------------------------------

def render_sheet(dpi: int = 300) -> np.ndarray:
    px = lambda cm: int(round(cm / 2.54 * dpi))  # noqa: E731
    img = np.full((px(PAGE_H_CM), px(PAGE_W_CM)), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    side = px(MARKER_CM)
    for mid, (cx, cy) in _MARKER_CENTERS_PAGE.items():
        m = cv2.aruco.generateImageMarker(dictionary, mid, side)
        x0, y0 = px(cx) - side // 2, px(cy) - side // 2
        img[y0 : y0 + side, x0 : x0 + side] = m
        cv2.putText(img, f"m{mid}", (x0, y0 + side + px(0.6)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    n = 1
    for y in _CROSS_Y:
        for x in _CROSS_X:
            c = (px(x), px(y))
            cv2.circle(img, c, px(0.6), 0, 3)
            cv2.line(img, (c[0] - px(0.9), c[1]), (c[0] + px(0.9), c[1]), 0, 2)
            cv2.line(img, (c[0], c[1] - px(0.9)), (c[0], c[1] + px(0.9)), 0, 2)
            cv2.putText(img, str(n), (c[0] + px(0.8), c[1] - px(0.8)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
            n += 1

    (x0, y0), (x1, _) = _SCALE_BAR
    cv2.line(img, (px(x0), px(y0)), (px(x1), px(y0)), 0, 4)
    for x in (x0, x1):
        cv2.line(img, (px(x), px(y0) - px(0.4)), (px(x), px(y0) + px(0.4)), 0, 4)
    cv2.putText(img, "this line must measure 10.0 cm", (px(x0) + px(0.8), px(y0) + px(1.2)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3)
    cv2.putText(img, "Print at 100% / Actual size. Tape flat.", (px(6.0), px(26.6)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, 0, 3)
    return img


def main() -> None:
    from PIL import Image

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "targets.pdf"
    Image.fromarray(render_sheet()).save(path, "PDF", resolution=300)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

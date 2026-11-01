"""Camera capture and the pixel <-> desk mapping (a plane homography).

The desk is flat and the camera is fixed, so one 3x3 homography maps any pixel
to a desk point (meters, world frame from deskbot.targets) and back.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import config
from .targets import ARUCO_DICT, OUT_DIR, marker_corners_world

CAMERA_FILE = OUT_DIR / "camera.json"
WIDTH, HEIGHT = 1920, 1080


def grab_frame(index: int = config.CAMERA_INDEX, settle_frames: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(f"Camera {index} unavailable (no permission, or wrong index).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    frame = None
    for _ in range(settle_frames):  # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.02)
    cap.release()
    if frame is None:
        raise RuntimeError("Camera opened but returned no frames.")
    return frame


def detect_markers(img: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco id -> 4x2 pixel corners."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


class DeskMap:
    """pixel <-> world (meters on the desk plane)."""

    def __init__(self, H: np.ndarray, image_size: tuple[int, int]):
        self.H = np.asarray(H, float)          # pixel -> world
        self.Hinv = np.linalg.inv(self.H)      # world -> pixel
        self.image_size = tuple(image_size)

    @classmethod
    def from_markers(cls, markers: dict[int, np.ndarray], image_size) -> tuple["DeskMap", float]:
        ids = [i for i in (0, 1, 2, 3) if i in markers]
        if len(ids) < 2:
            raise ValueError(f"Need at least 2 of markers 0-3 in view, found {sorted(markers)}.")
        px = np.vstack([markers[i] for i in ids])
        world = np.vstack([marker_corners_world(i) for i in ids])
        H, _ = cv2.findHomography(px, world, 0)
        m = cls(H, image_size)
        err_mm = float(np.max(np.linalg.norm(m.to_world(px) - world, axis=1)) * 1000)
        return m, err_mm

    def to_world(self, px) -> np.ndarray:
        p = np.atleast_2d(np.asarray(px, float))
        w = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.H).reshape(-1, 2)
        return w if np.ndim(px) > 1 else w[0]

    def to_pixel(self, xy) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xy, float))
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.Hinv).reshape(-1, 2)
        return q if np.ndim(xy) > 1 else q[0]

    def save(self, path: Path = CAMERA_FILE) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"H": self.H.tolist(), "image_size": list(self.image_size),
                                    "camera_index": config.CAMERA_INDEX}, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = CAMERA_FILE) -> "DeskMap":
        if not path.exists():
            raise SystemExit("No camera calibration yet. Run: python -m deskbot.calib_camera")
        d = json.loads(path.read_text())
        return cls(np.array(d["H"]), d["image_size"])


def draw_grid(img: np.ndarray, desk: DeskMap, step: float = 0.05, span=(-0.6, 0.8)) -> np.ndarray:
    """Overlay a world grid (every `step` m) so distortion/misfit is easy to see."""
    out = img.copy()
    h, w = out.shape[:2]
    vals = np.arange(span[0], span[1] + 1e-9, step)
    for v in vals:
        for line in (np.c_[np.full(200, v), np.linspace(*span, 200)],
                     np.c_[np.linspace(*span, 200), np.full(200, v)]):
            pts = desk.to_pixel(line)
            ok = (pts[:, 0] > -w) & (pts[:, 0] < 2 * w) & (pts[:, 1] > -h) & (pts[:, 1] < 2 * h)
            if ok.sum() > 1:
                color = (0, 0, 255) if abs(v) < 1e-9 else (0, 200, 255)
                cv2.polylines(out, [pts[ok].astype(np.int32)], False, color, 1, cv2.LINE_AA)
    return out

"""Save one frame from every camera so we can find the C920 and check its view.

    python -m deskbot.snap          # all cameras  -> snapshots/cam<N>.jpg
    python -m deskbot.snap 1        # just camera 1 -> snapshots/cam1.jpg
"""

import sys
import time
from pathlib import Path

import cv2

OUT = Path(__file__).resolve().parent.parent / "snapshots"


def grab(idx: int) -> Path | None:
    cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    frame = None
    for _ in range(30):  # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
        time.sleep(0.03)
    cap.release()
    if frame is None:
        return None
    OUT.mkdir(exist_ok=True)
    path = OUT / f"cam{idx}.jpg"
    cv2.imwrite(str(path), frame)
    return path


def main() -> None:
    indices = [int(sys.argv[1])] if len(sys.argv) > 1 else range(4)
    for idx in indices:
        path = grab(idx)
        print(f"camera {idx}: {path or 'not available'}")


if __name__ == "__main__":
    main()

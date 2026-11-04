"""Hardware settings for this desk.

Machine-specific values (serial ports, camera index) come from the project's `.env`
file so they never end up in git. Copy `.env.example` to `.env` and fill them in.
Find your ports with `lerobot-find-port`, and the camera index with `python -m deskbot.snap`.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional; plain environment variables also work
    pass

FOLLOWER_PORT = os.environ.get("GOOFE_FOLLOWER_PORT", "/dev/tty.usbmodem_FOLLOWER")
LEADER_PORT = os.environ.get("GOOFE_LEADER_PORT", "/dev/tty.usbmodem_LEADER")
ROBOT_ID = os.environ.get("GOOFE_ROBOT_ID", "goof_e")
LEADER_ID = os.environ.get("GOOFE_LEADER_ID", "goof_e_leader")

# Top-down webcam (a Logitech C920 on a boom arm). The index can shift when other cameras
# (e.g. iPhone Continuity Camera) connect; re-run `python -m deskbot.snap` to check.
CAMERA_INDEX = int(os.environ.get("GOOFE_CAMERA_INDEX", "0"))

# Named joint poses taught with the leader arm (see deskbot/teach.py).
POSES_FILE = ROOT / "poses.json"

# Safety
MAX_SPEED_DPS = 90.0     # fastest any joint may move, degrees/second
LIMIT_MARGIN_DEG = 3.0   # stay this far inside the calibrated joint range
CONTROL_HZ = 50

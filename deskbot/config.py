"""Hardware settings for this desk. Edit here, not in the code."""

from pathlib import Path

FOLLOWER_PORT = "/dev/tty.usbmodem5B3D0492451"
LEADER_PORT = "/dev/tty.usbmodem5B610354531"
ROBOT_ID = "goof_e"
LEADER_ID = "goof_e_leader"

# Logitech C920 (top-down). On 2026-10-31: 0 = C920, 1 = iPhone Continuity (black), 2 = FaceTime.
# The order can shift if the iPhone connects/disconnects; re-run `python -m deskbot.snap` to check.
CAMERA_INDEX = 0

# Named joint poses taught with the leader arm (see deskbot/teach.py).
POSES_FILE = Path(__file__).resolve().parent.parent / "poses.json"

# Safety
MAX_SPEED_DPS = 90.0     # fastest any joint may move, degrees/second
LIMIT_MARGIN_DEG = 3.0   # stay this far inside the calibrated joint range
CONTROL_HZ = 50

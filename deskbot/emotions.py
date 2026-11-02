"""Expressive gestures built from taught poses (no joint-sign guessing).

    python -m deskbot.emote happy
    python -m deskbot.emote sad

Needs the poses 'neutral', 'happy_up' and 'sad_bow' (see deskbot/teach.py).
"""

import time

from .arm import Arm
from .poses import require_pose

CUE_LEAD_S = 0.08  # fire the sound cue this much before the motion: covers the browser's playback start
CLOSED = 3.0  # gripper value for "closed" (a little above 0 so the jaws don't stall against each other)


def _cue(on_cue) -> None:
    if on_cue:
        on_cue()
        time.sleep(CUE_LEAD_S)


def happy(arm: Arm, bobs: int = 2, on_cue=None) -> None:
    """Bob the gripper up and down with the elbow, jaws opening on the way up.
    on_cue() fires just as the bobbing starts (for the "Yayy!")."""
    neutral = require_pose("neutral")
    up = {**require_pose("happy_up", "happy"), "gripper": 70.0}
    down = {**neutral, "gripper": 15.0}
    arm.move_to(neutral, speed=60)
    _cue(on_cue)
    for _ in range(bobs):
        arm.move_to(up, duration=0.35)
        arm.move_to(down, duration=0.35)
    arm.move_to(neutral, duration=0.4)


def sad(arm: Arm, on_cue=None) -> None:
    """Close the gripper, bow slowly, hang there, then straighten up.
    on_cue() fires just as the gripper starts closing (for the "Aww…")."""
    neutral = require_pose("neutral")
    bow = {**require_pose("sad_bow", "sad"), "gripper": CLOSED}
    arm.move_to(neutral, speed=60)
    _cue(on_cue)
    arm.move_to({"gripper": CLOSED}, duration=0.6)
    arm.move_to(bow, duration=2.2)
    time.sleep(1.5)
    arm.move_to({**neutral, "gripper": CLOSED}, duration=1.6)


EMOTIONS = {"happy": happy, "sad": sad}

"""Expressive gestures built from taught poses (no joint-sign guessing).

    python -m deskbot.emote happy
    python -m deskbot.emote sad

Needs the poses 'neutral', 'happy_up' and 'sad_bow' (see deskbot/teach.py).
"""

import time

from .arm import Arm
from .poses import require_pose

CLOSED = 3.0  # gripper value for "closed" (a little above 0 so the jaws don't stall against each other)


def happy(arm: Arm, bobs: int = 2) -> None:
    """Bob the gripper up and down with the elbow, jaws opening on the way up."""
    neutral = require_pose("neutral")
    up = {**require_pose("happy_up", "happy"), "gripper": 70.0}
    down = {**neutral, "gripper": 15.0}
    arm.move_to(neutral, speed=60)
    for _ in range(bobs):
        arm.move_to(up, duration=0.35)
        arm.move_to(down, duration=0.35)
    arm.move_to(neutral, duration=0.4)


def sad(arm: Arm) -> None:
    """Close the gripper, bow slowly, hang there, then straighten up."""
    neutral = require_pose("neutral")
    bow = {**require_pose("sad_bow", "sad"), "gripper": CLOSED}
    arm.move_to(neutral, speed=60)
    arm.move_to({"gripper": CLOSED}, duration=0.6)
    arm.move_to(bow, duration=2.2)
    time.sleep(1.5)
    arm.move_to({**neutral, "gripper": CLOSED}, duration=1.6)


EMOTIONS = {"happy": happy, "sad": sad}

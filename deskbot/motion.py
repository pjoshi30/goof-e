"""Cartesian moves on the desk built on IK: go above a point, descend, come back up."""

import time

from .arm import Arm
from .kinematics import ArmModel
from .poses import load_poses

SAFE_Z = 0.15   # m: travel height between points
HOVER_Z = 0.07  # m: pause height above a point before descending


def plan_points(arm: Arm, model: ArmModel, points, roll: float = 0.0):
    """IK for a list of xyz points, each seeded from the previous. None if any is unreachable."""
    seed = arm.read()
    out = []
    for xyz in points:
        q = model.ik(xyz, arm.limits, roll=roll, seed=seed)
        if q is None:
            return None
        out.append(q)
        seed = q
    return out


def ensure_clear(arm: Arm, model: ArmModel) -> None:
    """If the tip is low (e.g. folded at rest), rise through 'neutral' before travelling."""
    if model.tip_position(arm.read())[2] < 0.08 and "neutral" in load_poses():
        arm.move_to(load_poses()["neutral"], speed=45)


def touch_point(arm: Arm, model: ArmModel, x: float, y: float, z: float = 0.01,
                dwell: float = 1.0) -> str | None:
    """Travel above (x, y), descend slowly to height z, pause, rise. Returns an error or None."""
    plan = plan_points(arm, model, [(x, y, SAFE_Z), (x, y, HOVER_Z), (x, y, z)])
    if plan is None:
        return f"({x*100:.1f}, {y*100:.1f}) cm is out of reach"
    high, hover, low = plan
    ensure_clear(arm, model)
    arm.move_to({"gripper": 3.0}, duration=0.5)
    arm.move_to(high, speed=45)
    arm.move_to(hover, speed=30)
    arm.move_to(low, speed=10)
    arm.settle(low)
    time.sleep(dwell)
    arm.move_to(hover, speed=20)
    return None

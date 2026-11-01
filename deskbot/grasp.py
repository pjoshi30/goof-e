"""Pick an object up from the desk and put it down somewhere else.

The SO-101 jaws aren't symmetric: the fixed jaw carries the tool tip, and the moving jaw
swings open on the gripper link's +x side (from the URDF: the jaw hinge sits at +x and turns
about the link's y axis). So to grab something we line +x up with the object's short side,
put the fixed jaw just beside it, open, descend, and close: the moving jaw sweeps the
object against the fixed jaw.
"""

import time

import numpy as np

from .arm import Arm
from .kinematics import ArmModel
from .motion import HOVER_Z, SAFE_Z, ensure_clear

GRASP_Z = 0.004          # jaw tips this high above the desk when closing, m. Low on purpose:
                         # closing above a round object's middle squeezes it out of the jaws.
PLACE_Z = 0.02           # let go this high above the desk, m
SIDE_MARGIN_MM = 7.0     # gap between the fixed jaw and the object as we descend
OPEN_EXTRA_MM = 18.0     # opening beyond (object + margin), to absorb position error
GRIP_SPAN_DEG = 132.7    # gripper 0..100 covers this many degrees (from the calibration range)
JAW_LEN_M = 0.07         # hinge to jaw tip, for converting opening angle <-> gap
CLOSED = 0.0
CLOSE_S = 2.0            # seconds to close the jaws: slow, so they ease onto fragile things
EMPTY_CLOSED = 1.0       # gripper reading when closed on nothing (jaws touch at ~0.99; measured 2026-10-31)
# Above this after closing = something is in the jaws. Kept just above empty on purpose: thin
# things (a bracket wall, a squeezed marker) read only ~1.5-4, and wrongly thinking we hold
# nothing is worse (we'd park carrying it) than wrongly thinking we hold something.
HELD_MIN = EMPTY_CLOSED + 0.5


def gripper_for_gap(gap_mm: float) -> float:
    gap = min(gap_mm / 1000, 1.9 * JAW_LEN_M)
    deg = np.degrees(2 * np.arcsin(gap / (2 * JAW_LEN_M)))
    return float(np.clip(deg / GRIP_SPAN_DEG * 100, 0, 100))


def _heading(model: ArmModel, q: dict) -> float:
    """World direction (deg) in which the moving jaw opens, projected onto the desk."""
    x = model.fk(q)[1][:, 0]
    return float(np.degrees(np.arctan2(x[1], x[0])))


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def plan_grasp(model: ArmModel, limits: dict, x: float, y: float, grasp_deg: float,
               width_mm: float, seed: dict | None = None):
    """-> (q_low, q_hover, q_high, opening) or None.

    grasp_deg is the direction the jaws should close along. Either way round works
    (moving jaw on one side or the other); we pick whichever needs less wrist roll."""
    centre = np.array([x, y])
    lo, hi = limits["wrist_roll"]
    best = None
    for flip in (0.0, 180.0):
        g = grasp_deg + flip
        u = np.array([np.cos(np.radians(g)), np.sin(np.radians(g))])
        tip = centre - (width_mm / 2 + SIDE_MARGIN_MM) / 1000 * u  # fixed jaw beside the object
        roll, q = 0.0, None
        for _ in range(4):
            q = model.ik((*tip, GRASP_Z), limits, roll=roll, tilts_deg=(0, 15, 30), seed=seed)
            if q is None:
                break
            miss = _wrap(g - _heading(model, q))
            if abs(miss) < 3:
                break
            # how heading responds to roll (sign depends on arm pose): measure it
            slope = _wrap(_heading(model, {**q, "wrist_roll": roll + 5}) - _heading(model, q)) / 5
            roll = float(np.clip(roll + miss / (slope if abs(slope) > 0.3 else 1.0), lo, hi))
        if q is None or abs(_wrap(g - _heading(model, q))) > 6:
            continue
        hover = model.ik((*tip, HOVER_Z), limits, roll=roll, seed=q)
        high = model.ik((*tip, SAFE_Z), limits, roll=roll, seed=hover)
        if hover is None or high is None:
            continue
        if best is None or abs(roll) < abs(best[0]):
            best = (roll, q, hover, high)
    if best is None:
        return None
    opening = gripper_for_gap(width_mm + SIDE_MARGIN_MM + OPEN_EXTRA_MM)
    return best[1], best[2], best[3], opening


def pick(arm: Arm, model: ArmModel, x: float, y: float, grasp_deg: float, width_mm: float) -> str | None:
    """Grab the object at (x, y). Returns an error message, or None if it's now held."""
    plan = plan_grasp(model, arm.limits, x, y, grasp_deg, width_mm, seed=arm.read())
    if plan is None:
        return f"can't line the gripper up at ({x*100:.1f}, {y*100:.1f}) cm"
    low, hover, high, opening = plan
    ensure_clear(arm, model)
    arm.move_to({**high, "gripper": CLOSED + 3}, speed=45)
    arm.move_to({**hover, "gripper": opening}, speed=30)
    arm.settle(hover)
    arm.move_to(low, speed=12)
    arm.settle(low)
    arm.move_to({"gripper": CLOSED}, duration=CLOSE_S)
    time.sleep(0.4)
    held = arm.read()["gripper"]
    if held < HELD_MIN:
        arm.move_to({**hover, "gripper": CLOSED}, speed=20)
        return "missed: the jaws closed on nothing"
    arm.move_to({**hover, "gripper": CLOSED}, speed=20)
    time.sleep(0.3)
    if arm.read()["gripper"] < HELD_MIN:
        return "it slipped out of the jaws while lifting"
    arm.move_to({**high, "gripper": CLOSED}, speed=30)
    return None


def _jaw_dir(model: ArmModel, q: dict) -> np.ndarray:
    """Unit vector on the desk from the fixed jaw towards the moving jaw."""
    x = model.fk(q)[1][:2, 0]
    return x / (np.linalg.norm(x) + 1e-9)


def place(arm: Arm, model: ArmModel, x: float, y: float, held_width_mm: float = 0.0) -> str | None:
    """Carry what's held so its centre lands on (x, y), set it down, let go, back off.

    The jaws clamp the object against the fixed jaw, so its centre sits half its width
    beside the tool tip (towards the moving jaw). Aim the tip that far short."""
    seed = arm.read()
    roll = seed["wrist_roll"]
    offset = held_width_mm / 2000
    target = np.array([x, y])
    tip, pts = target, None
    for _ in range(3):  # the jaw direction depends on the arm pose, so refine a couple of times
        pts = [model.ik((*tip, z), arm.limits, roll=roll, seed=seed) for z in (SAFE_Z, HOVER_Z, PLACE_Z)]
        if any(p is None for p in pts):
            return f"({x*100:.1f}, {y*100:.1f}) cm is out of reach"
        tip = target - offset * _jaw_dir(model, pts[2])
    high, hover, low = pts
    held = {"gripper": CLOSED}
    arm.move_to({**high, **held}, speed=40)
    arm.move_to({**hover, **held}, speed=25)
    arm.move_to({**low, **held}, speed=12)
    arm.settle(low)
    arm.move_to({"gripper": 45.0}, duration=0.6)
    time.sleep(0.3)
    arm.move_to(hover, speed=25)
    arm.move_to({**high, "gripper": CLOSED + 3}, speed=40)
    return None

"""Hold the gripper tip a set height above a printed target, so you can measure the gap.

    python -m deskbot.hover 5 20      # 20 mm above target 5
    python -m deskbot.hover m1 20

Press Enter to move to the next height or target you type (e.g. "3 20"), or "q" to park.
"""

import sys

from .arm import Arm
from .kinematics import ArmModel
from .motion import HOVER_Z, SAFE_Z, ensure_clear, plan_points
from .targets import touch_targets


def hover(arm: Arm, model: ArmModel, name: str, z_mm: float) -> None:
    x, y = touch_targets()[name]
    plan = plan_points(arm, model, [(x, y, SAFE_Z), (x, y, HOVER_Z), (x, y, z_mm / 1000)])
    if plan is None:
        print(f"  {name} is out of reach")
        return
    ensure_clear(arm, model)
    arm.move_to({"gripper": 3.0}, duration=0.5)
    arm.move_to(plan[0], speed=45)
    arm.move_to(plan[1], speed=30)
    arm.move_to(plan[2], speed=10)
    arm.settle(plan[2])
    print(f"  holding {z_mm:.0f} mm above {name}. Measure from the desk to the lowest point of the jaws.")


def rise_from(arm: Arm, model: ArmModel, name: str) -> None:
    """Back up to hover height above the target we're at, before travelling elsewhere."""
    if name in touch_targets():
        x, y = touch_targets()[name]
        up = plan_points(arm, model, [(x, y, HOVER_Z)])
        if up:
            arm.move_to(up[0], speed=20)


def main() -> None:
    args = sys.argv[1:] or ["5", "20"]
    model = ArmModel.load()
    with Arm() as arm:
        while True:
            name, z = args[0], float(args[1]) if len(args) > 1 else 20.0
            if name not in touch_targets():
                print(f"  unknown target '{name}'")
            else:
                hover(arm, model, name, z)
            line = input("next (e.g. '3 20'), or q: ").strip()
            if line == "q":
                rise_from(arm, model, name)
                break
            if line:
                args = line.split()
            rise_from(arm, model, name)


if __name__ == "__main__":
    main()

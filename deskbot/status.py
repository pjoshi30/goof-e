"""Print the follower's joint positions and safe limits. Never moves the arm or changes torque.

    python -m deskbot.status
"""

from .arm import JOINTS, Arm


def main() -> None:
    arm = Arm().connect(torque=False)
    try:
        pose = arm.read()
        torque = arm.robot.bus.sync_read("Torque_Enable", normalize=False)
        print(f"{'joint':<14}{'now':>9}{'min':>9}{'max':>9}  torque")
        for j in JOINTS:
            lo, hi = arm.limits[j]
            print(f"{j:<14}{pose[j]:9.1f}{lo:9.1f}{hi:9.1f}  {'on' if torque[j] else 'off'}")
    finally:
        arm.robot.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()

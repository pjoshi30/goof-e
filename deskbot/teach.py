"""Teach named poses by moving the leader arm; the follower mirrors it.

    python -m deskbot.teach

Move the leader, then type a name and press Enter to save the follower's pose.
Commands: 'list' shows saved poses, 'q' quits (the follower then parks at 'rest').

Poses the code uses:
  rest      folded, the way the arm sits when powered off
  neutral   upright and "looking" at the desk; the home pose for emotions
  happy     like neutral, but elbow and wrist raised (top of the happy bob)
  sad       bowed forward and down, head hanging (the sad pose)
"""

from .arm import Arm
from .poses import load_poses, save_pose
from .teleop import run_teleop


def on_command(cmd: str, arm: Arm) -> bool:
    if cmd == "q":
        return False
    if cmd == "list":
        for name, pose in load_poses().items():
            print(f"  {name}: {pose}")
    else:
        save_pose(cmd, arm.read())
        print(f"  saved '{cmd}'")
    return True


def main() -> None:
    run_teleop(on_command, intro=__doc__)


if __name__ == "__main__":
    main()

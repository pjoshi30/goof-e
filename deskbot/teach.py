"""Teach named poses by moving the leader arm; the follower mirrors it.

    python -m deskbot.teach

Move the leader, then type a name and press Enter to save the follower's pose.
Commands: 'list' shows saved poses, 'q' quits (the follower then parks at 'rest').

Teach these first:
  rest      folded, the way the arm sits when powered off
  neutral   upright and "looking" at the desk; the home pose for emotions
  happy_up  like neutral, but elbow and wrist raised (top of the happy bob)
  sad_bow   bowed forward and down, head hanging (the sad pose)
"""

import queue
import sys
import threading
import time

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from . import config
from .arm import Arm
from .poses import load_poses, save_pose


def _read_lines(q: "queue.Queue[str]") -> None:
    for line in sys.stdin:
        q.put(line.strip())


def main() -> None:
    leader = SO101Leader(SO101LeaderConfig(port=config.LEADER_PORT, id=config.LEADER_ID))
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        raise SystemExit("Leader calibration doesn't match its file. Recalibrate the leader first.")

    def leader_pose() -> dict[str, float]:
        return {k.removesuffix(".pos"): v for k, v in leader.get_action().items()}

    try:
        with Arm() as arm:
            print("Slowly matching the follower to the leader…")
            arm.move_to(leader_pose(), speed=30)
            print(__doc__)
            lines: queue.Queue[str] = queue.Queue()
            threading.Thread(target=_read_lines, args=(lines,), daemon=True).start()

            dt = 1.0 / 30
            while True:
                t0 = time.perf_counter()
                target = arm.clamp(leader_pose())
                arm.robot.send_action({f"{j}.pos": v for j, v in target.items()})

                try:
                    cmd = lines.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd == "q":
                    break
                if cmd == "list":
                    for name, pose in load_poses().items():
                        print(f"  {name}: {pose}")
                elif cmd:
                    save_pose(cmd, arm.read())
                    print(f"  saved '{cmd}'")

                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    finally:
        leader.disconnect()


if __name__ == "__main__":
    main()

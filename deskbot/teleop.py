"""Leader -> follower mirroring with typed commands (shared by teach and calib_arm)."""

import queue
import sys
import threading
import time
from collections.abc import Callable

from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

from . import config
from .arm import Arm


def run_teleop(on_command: Callable[[str, Arm], bool], intro: str = "") -> None:
    """Mirror the leader onto the follower at 30 Hz. Each typed line goes to
    on_command(line, arm); return False from it to stop. Ctrl+C stops safely."""
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
            if intro:
                print(intro)
            lines: queue.Queue[str] = queue.Queue()

            def read_lines() -> None:
                for line in sys.stdin:
                    lines.put(line.strip())

            threading.Thread(target=read_lines, daemon=True).start()
            dt = 1.0 / 30
            while True:
                t0 = time.perf_counter()
                target = arm.clamp(leader_pose())
                arm.robot.send_action({f"{j}.pos": v for j, v in target.items()})
                try:
                    cmd = lines.get_nowait()
                except queue.Empty:
                    cmd = None
                if cmd and not on_command(cmd, arm):
                    break
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    finally:
        leader.disconnect()

"""Safe, smooth control of the SO-101 follower on top of LeRobot.

Units: body joints in degrees, gripper 0 (closed) .. 100 (open).

Safety behaviour:
- Torque is enabled with the goal set to the current position, so connecting never jumps.
- Every target is clamped inside the calibrated range (LeRobot does not clamp in degree mode).
- Joint speed is capped at MAX_SPEED_DPS.
- Ctrl+C freezes the arm where it is, then parks it at the 'rest' pose and relaxes.
  A second Ctrl+C while parking leaves it holding position (torque on).
"""

import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from . import config
from .poses import Pose, load_poses

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_SPEED = 200.0  # gripper units/second (full open<->close in ~0.5 s)


def _min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Arm:
    def __init__(self, port: str = config.FOLLOWER_PORT, robot_id: str = config.ROBOT_ID):
        cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        self.robot = SO101Follower(cfg)
        self.limits: dict[str, tuple[float, float]] = {}

    # ---- lifecycle -------------------------------------------------------

    def connect(self, torque: bool = True) -> "Arm":
        bus = self.robot.bus
        bus.connect()
        if not self.robot.is_calibrated:
            bus.disconnect(disable_torque=False)
            raise RuntimeError("Motor calibration doesn't match the calibration file. Recalibrate first.")
        self._compute_limits()
        if torque:
            # Make the goal equal to where the arm is, so enabling torque can't cause a jump.
            present = bus.sync_read("Present_Position", normalize=False)
            bus.sync_write("Goal_Position", present, normalize=False)
            self.robot.configure()  # PID + gripper current limits; re-enables torque
        return self

    def _compute_limits(self) -> None:
        m = config.LIMIT_MARGIN_DEG
        for j in JOINTS:
            if j == "gripper":
                self.limits[j] = (0.0, 100.0)
                continue
            cal = self.robot.bus.calibration[j]
            mid = (cal.range_min + cal.range_max) / 2
            lo = (cal.range_min - mid) * 360 / 4095 + m
            hi = (cal.range_max - mid) * 360 / 4095 - m
            self.limits[j] = (lo, hi)

    def __enter__(self) -> "Arm":
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        if exc_type is KeyboardInterrupt:
            print("\nStopped — holding position.")
            self.hold()
        self.shutdown()
        return exc_type is KeyboardInterrupt  # swallow Ctrl+C after a safe stop

    def shutdown(self) -> None:
        """Park at 'rest' if taught, then relax. Otherwise keep holding (torque on)."""
        rest = load_poses().get("rest")
        if rest is None:
            print("No 'rest' pose taught — leaving torque ON so the arm doesn't drop.")
            self.robot.bus.disconnect(disable_torque=False)
            return
        try:
            print("Parking at 'rest'…")
            self.move_to(rest, speed=40)
            time.sleep(0.3)
            self.robot.bus.disconnect(disable_torque=True)
        except KeyboardInterrupt:
            print("\nParking interrupted — holding position, torque left ON.")
            self.hold()
            self.robot.bus.disconnect(disable_torque=False)

    # ---- state -----------------------------------------------------------

    def read(self) -> Pose:
        pos = self.robot.bus.sync_read("Present_Position")
        return {j: float(pos[j]) for j in JOINTS}

    def clamp(self, pose: Pose) -> Pose:
        out = {}
        for j, v in pose.items():
            lo, hi = self.limits[j]
            out[j] = min(hi, max(lo, v))
        return out

    def hold(self) -> None:
        self.robot.bus.sync_write("Goal_Position", self.read())

    def relax(self) -> None:
        self.robot.bus.disable_torque()

    # ---- motion ----------------------------------------------------------

    def move_to(self, target: Pose, duration: float | None = None, speed: float | None = None) -> None:
        """Smoothly move to `target` (joints you leave out stay put).

        duration: seconds; stretched if it would exceed the speed cap.
        speed: degrees/second for the slowest-allowed estimate (default MAX_SPEED_DPS).
        """
        start = self.read()
        goal = self.clamp({**start, **{k: v for k, v in target.items() if k in JOINTS}})

        cap = min(speed or config.MAX_SPEED_DPS, config.MAX_SPEED_DPS)
        min_t = 0.0
        for j in JOINTS:
            delta = abs(goal[j] - start[j])
            # min-jerk peak speed is 1.875x the average speed
            min_t = max(min_t, 1.875 * delta / (GRIPPER_SPEED if j == "gripper" else cap))
        t_total = max(duration or 0.0, min_t, 0.1)

        dt = 1.0 / config.CONTROL_HZ
        t0 = time.perf_counter()
        while True:
            t = time.perf_counter() - t0
            s = _min_jerk(min(1.0, t / t_total))
            cmd = {f"{j}.pos": start[j] + (goal[j] - start[j]) * s for j in JOINTS}
            self.robot.send_action(cmd)
            if t >= t_total:
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - t0 - t)))

    def settle(self, target: Pose, rounds: int = 4, tol_deg: float = 0.3, max_fix_deg: float = 6.0) -> None:
        """Nudge the arm onto `target` after a move. The servos stop a little short under the
        arm's weight (worst when reaching far), so command past the target by the measured shortfall."""
        joints = [j for j in target if j in JOINTS and j != "gripper"]
        cmd = {j: target[j] for j in joints}
        for _ in range(rounds):
            time.sleep(0.25)
            now = self.read()
            err = {j: target[j] - now[j] for j in joints}
            if max(abs(e) for e in err.values()) < tol_deg:
                break
            for j in joints:
                cmd[j] = target[j] + max(-max_fix_deg, min(max_fix_deg, cmd[j] - target[j] + err[j]))
            self.robot.send_action({f"{j}.pos": v for j, v in self.clamp(cmd).items()})

    def play(self, keyframes: list[tuple[Pose, float]]) -> None:
        """Run a list of (pose, seconds) moves back to back."""
        for pose, seconds in keyframes:
            self.move_to(pose, duration=seconds)


def joint_limits() -> dict[str, tuple[float, float]]:
    """Safe joint ranges from the calibration file, without opening the serial port."""
    arm = Arm()
    arm._compute_limits()
    return arm.limits


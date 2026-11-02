"""Safe, smooth control of the SO-101 follower on top of LeRobot.

Units: body joints in degrees, gripper 0 (closed) .. 100 (open).

Safety behaviour:
- Torque is enabled with the goal set to the current position, so connecting never jumps.
- Every target is clamped inside the calibrated range (LeRobot does not clamp in degree mode).
- Joint speed is capped at MAX_SPEED_DPS.
- Ctrl+C freezes the arm where it is, then parks it at the 'rest' pose and relaxes.
  A second Ctrl+C while parking leaves it holding position (torque on).
"""

import threading
import time
from contextlib import contextmanager

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

from . import config
from .poses import Pose, load_poses

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
GRIPPER_SPEED = 200.0  # gripper units/second (full open<->close in ~0.5 s)


class Stopped(Exception):
    """Raised inside a move when someone pressed Stop; the arm is already holding still."""


def interruptible(stop_event: threading.Event, fn, *args, **kwargs):
    """Run a slow call (an API request, a camera grab) so Stop takes effect at once:
    on Stop, raise Stopped right away and let the abandoned call finish in the background."""
    box: dict = {}

    def work():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # hand any error back to the caller
            box["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        if stop_event.is_set():
            raise Stopped()
        t.join(0.05)
    if "error" in box:
        raise box["error"]
    return box["value"]


def _min_jerk(s: float) -> float:
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Arm:
    def __init__(self, port: str = config.FOLLOWER_PORT, robot_id: str = config.ROBOT_ID):
        cfg = SO101FollowerConfig(port=port, id=robot_id, use_degrees=True)
        self.robot = SO101Follower(cfg)
        self.limits: dict[str, tuple[float, float]] = {}
        self.stop_event = threading.Event()   # set from another thread to halt the current move
        # on_motion(True) when the arm starts moving, on_motion(False) once it has been still a while
        self.on_motion = None
        self._moving, self._last_move = False, 0.0

    # ---- lifecycle -------------------------------------------------------

    def connect(self, torque: bool = True) -> "Arm":
        bus = self.robot.bus
        bus.connect()
        if not self.robot.is_calibrated:
            bus.disconnect(disable_torque=False)
            raise RuntimeError("Motor calibration doesn't match the calibration file. Recalibrate first.")
        self._compute_limits()
        if torque:
            # The serial bus occasionally garbles one reply right after opening
            # ("Incorrect status packet"); retry the setup a couple of times before giving up.
            for attempt in range(3):
                try:
                    # Make the goal equal to where the arm is, so enabling torque can't cause a jump.
                    present = bus.sync_read("Present_Position", normalize=False)
                    bus.sync_write("Goal_Position", present, normalize=False)
                    self.robot.configure()  # PID + gripper current limits; re-enables torque
                    break
                except ConnectionError as e:
                    if attempt == 2:
                        bus.disconnect(disable_torque=False)
                        raise ConnectionError(f"{e}\nThe arm isn't answering reliably: check its power "
                                              "supply and the USB / motor cables.") from e
                    print(f"  arm didn't answer cleanly ({e}); retrying…")
                    time.sleep(0.3)
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
        self.stop_event.clear()  # a pending Stop shouldn't block parking
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
            if self.stop_event.is_set():
                self.hold()
                raise Stopped()
            t = time.perf_counter() - t0
            self._moved()
            s = _min_jerk(min(1.0, t / t_total))
            cmd = {f"{j}.pos": start[j] + (goal[j] - start[j]) * s for j in JOINTS}
            self.robot.send_action(cmd)
            if t >= t_total:
                break
            time.sleep(max(0.0, dt - (time.perf_counter() - t0 - t)))

    MOTION_IDLE_S = 2.0   # this long without a move = "stopped moving"

    def _moved(self) -> None:
        """Note that the arm is moving now; tell the listener on start, and (from a watcher
        thread) once it has been still for MOTION_IDLE_S."""
        self._last_move = time.monotonic()
        if self._moving or self.on_motion is None:
            return
        self._moving = True
        self.on_motion(True)

        def watch():
            while self._busy or time.monotonic() - self._last_move < self.MOTION_IDLE_S:
                time.sleep(0.2)
            self._moving = False
            if self.on_motion:
                self.on_motion(False)

        threading.Thread(target=watch, daemon=True).start()

    _busy = 0   # > 0 while inside busy(): counts as moving even during pauses

    @contextmanager
    def busy(self):
        """Treat a whole task (e.g. pick, camera check, place) as one motion, so pauses in it
        (like the camera checking the grab) don't count as the arm having stopped."""
        self._busy += 1
        try:
            yield
        finally:
            self._busy -= 1
            self._last_move = time.monotonic()   # then the usual idle wait before "stopped"

    def settle(self, target: Pose, rounds: int = 4, tol_deg: float = 0.3, max_fix_deg: float = 6.0) -> None:
        """Nudge the arm onto `target` after a move. The servos stop a little short under the
        arm's weight (worst when reaching far), so command past the target by the measured shortfall."""
        joints = [j for j in target if j in JOINTS and j != "gripper"]
        cmd = {j: target[j] for j in joints}
        for _ in range(rounds):
            time.sleep(0.25)
            if self.stop_event.is_set():
                self.hold()
                raise Stopped()
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


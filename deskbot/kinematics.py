"""SO-101 forward/inverse kinematics and the touch-calibration fit.

Geometry comes from TheRobotStudio's so101_new_calib.urdf, whose joint zeros match
LeRobot's calibration (joint value in degrees == URDF angle). What we don't know is
where the arm sits on the desk and the small per-joint zero errors, so those are
fitted from "touch" samples: joint readings taken while the gripper tip touches
known points on the printed sheet.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .targets import OUT_DIR

ARM_FILE = OUT_DIR / "arm.json"
CHAIN_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
IK_JOINTS = CHAIN_JOINTS[:4]  # roll is chosen separately (object orientation)

# (xyz, rpy) of each revolute joint relative to its parent link; all rotate about local z.
_URDF_JOINTS = [
    ((0.0388353, -8.97657e-09, 0.0624), (np.pi, 0.0, -np.pi)),             # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-np.pi / 2, -np.pi / 2, 0.0)),    # shoulder_lift
    ((-0.11257, -0.028, 0.0), (0.0, 0.0, np.pi / 2)),                     # elbow_flex
    ((-0.1349, 0.0052, 0.0), (0.0, 0.0, -np.pi / 2)),                     # wrist_flex
    ((0.0, -0.0611, 0.0181), (np.pi / 2, 0.0486795, np.pi)),              # wrist_roll
]
# Nominal tool point in the gripper link (URDF gripper_frame); the fit refines it to the jaw tip.
NOMINAL_TIP = (-0.0079, -0.000218, -0.0981)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _mul_rz(R, a):
    """R @ _rz(a) without building the matrix (fk runs ~100k times per IK solve)."""
    c, s = math.cos(a), math.sin(a)
    out = R.copy()
    out[:, 0] = R[:, 0] * c + R[:, 1] * s
    out[:, 1] = R[:, 1] * c - R[:, 0] * s
    return out


_FIXED = [(np.array(xyz), _rpy(*rpy)) for xyz, rpy in _URDF_JOINTS]


@dataclass
class ArmModel:
    base_xyz: list = field(default_factory=lambda: [0.0, 0.0, 0.0])  # base_link origin, world m
    base_yaw: float = 0.0                                              # rad
    offsets_deg: list = field(default_factory=lambda: [0.0] * 5)       # added to each joint reading
    signs: list = field(default_factory=lambda: [1] * 5)
    tip: list = field(default_factory=lambda: list(NOMINAL_TIP))       # tool point in gripper link, m
    rms_mm: float | None = None

    # ---- forward kinematics --------------------------------------------------

    def fk(self, q: dict) -> tuple[np.ndarray, np.ndarray]:
        """Joint readings (deg) -> (tip position world m, gripper rotation world)."""
        R = _rz(self.base_yaw)
        p = np.array(self.base_xyz, float)
        for k, (xyz, Rf) in enumerate(_FIXED):
            p = p + R @ xyz
            ang = math.radians(self.signs[k] * q.get(CHAIN_JOINTS[k], 0.0) + self.offsets_deg[k])
            R = _mul_rz(R @ Rf, ang)
        return p + R @ np.asarray(self.tip, float), R

    def tip_position(self, q: dict) -> np.ndarray:
        return self.fk(q)[0]

    @staticmethod
    def approach(R: np.ndarray) -> np.ndarray:
        """Direction the gripper points (its jaws extend along -z of the gripper link)."""
        return -R[:, 2]

    # ---- inverse kinematics --------------------------------------------------

    def ik(self, xyz, limits: dict, roll: float = 0.0, tilts_deg=(0, 15, 30, 45),
           tol_mm: float = 2.0, seed: dict | None = None) -> dict | None:
        """Joint targets (deg) putting the tip at xyz with the gripper pointing down.

        Tries straight down first, then tilts the approach outward if that's out of reach.
        Returns None if no solution is within joint limits.
        """
        target = np.asarray(xyz, float)
        d = target[:2] - np.array(self.base_xyz[:2])
        out = d / (np.linalg.norm(d) + 1e-9)
        for tilt in tilts_deg:
            t = np.deg2rad(tilt)
            a_des = np.array([out[0] * np.sin(t), out[1] * np.sin(t), -np.cos(t)])
            sol = self._solve(target, a_des, limits, roll, tol_mm, seed)
            if sol is not None:
                return sol
        return None

    def _solve(self, target, a_des, limits, roll, tol_mm, seed):
        lo = np.array([limits[j][0] for j in IK_JOINTS])
        hi = np.array([limits[j][1] for j in IK_JOINTS])

        def err(x):
            q = dict(zip(IK_JOINTS, x), wrist_roll=roll)
            p, R = self.fk(q)
            return np.concatenate([(p - target), 0.05 * (self.approach(R) - a_des)])

        # Pan seed: the pan angle whose tip azimuth best matches the target.
        mid = (lo + hi) / 2
        pans = np.linspace(lo[0], hi[0], 25)
        az_t = np.arctan2(*(target[:2] - self.base_xyz[:2])[::-1])

        def az(pan):
            p = self.tip_position(dict(zip(IK_JOINTS, [pan, *mid[1:]]), wrist_roll=roll))
            return np.arctan2(*(p[:2] - self.base_xyz[:2])[::-1])

        pan0 = min(pans, key=lambda a: abs(np.angle(np.exp(1j * (az(a) - az_t)))))
        seeds = [np.array([seed[j] for j in IK_JOINTS])] if seed else []
        for lift in (-60, -20, 20):
            for elbow in (60, 20, -20):
                for wf in (60, 20):
                    seeds.append(np.array([pan0, lift, elbow, wf], float))

        best = None
        for i, x in enumerate(seeds):
            x = np.clip(x, lo, hi)
            lam = 1e-2
            e = err(x)
            for _ in range(60):
                J = np.empty((len(e), 4))
                for k in range(4):
                    dx = np.zeros(4)
                    dx[k] = 1e-3
                    J[:, k] = (err(x + dx) - e) / 1e-3
                step = np.linalg.solve(J.T @ J + lam * np.eye(4), -J.T @ e)
                x_new = np.clip(x + step, lo, hi)
                e_new = err(x_new)
                if e_new @ e_new < e @ e:
                    x, e, lam = x_new, e_new, lam * 0.5
                else:
                    lam *= 4
                if np.linalg.norm(e[:3]) < 1e-4 or lam > 1e3:
                    break
            pos_mm = np.linalg.norm(e[:3]) * 1000
            ang_deg = np.degrees(np.linalg.norm(e[3:]) / 0.05)
            if pos_mm < tol_mm and ang_deg < 5 and (best is None or pos_mm < best[0]):
                best = (pos_mm, x)
                if seed and i == 0 and pos_mm < 0.5:
                    break   # the current pose led straight there: keep it (no needless flips)
        if best is None:
            return None
        return {**dict(zip(IK_JOINTS, map(float, best[1]))), "wrist_roll": float(roll)}

    # ---- persistence ----------------------------------------------------------

    def save(self, path: Path = ARM_FILE) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = ARM_FILE) -> "ArmModel":
        if not path.exists():
            raise SystemExit("No arm calibration yet. Run: python -m deskbot.calib_arm")
        return cls(**json.loads(path.read_text()))


# ---- fitting -----------------------------------------------------------------

_FREE_OFFSETS = (1, 2, 3)  # pan offset is absorbed by base_yaw; roll barely moves the tip


def _unpack(p, signs):
    offs = [0.0] * 5
    for k, i in enumerate(_FREE_OFFSETS):
        offs[i] = p[4 + k]
    return ArmModel(base_xyz=list(p[0:3]), base_yaw=float(p[3]), offsets_deg=offs,
                    signs=list(signs), tip=list(p[7:10]))


def _residuals(p, signs, samples):
    m = _unpack(p, signs)
    r = [m.tip_position(q) - np.array([x, y, 0.0]) for q, (x, y) in samples]
    priors = [p[2] * 1.0,                                  # base sits on the desk
              *(p[4:7] * 0.0003),                          # joint offsets ~0 (10 deg ~ 3 mm)
              *((p[7:10] - np.array(NOMINAL_TIP)) * 1.0)]  # tip near the nominal tool point
    return np.concatenate([np.concatenate(r), priors])


def _kabsch_init(signs, samples):
    m = ArmModel(signs=list(signs))
    P = np.array([m.tip_position(q) for q, _ in samples])
    W = np.array([xy for _, xy in samples])
    Pc, Wc = P[:, :2] - P[:, :2].mean(0), W - W.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Wc)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:  # reflection: these signs can't match; fall back to best rotation
        Vt[-1] *= -1
        R = (U @ Vt).T
    yaw = np.arctan2(R[1, 0], R[0, 0])
    t = W.mean(0) - R @ P[:, :2].mean(0)
    return np.array([t[0], t[1], -P[:, 2].mean(), yaw, 0, 0, 0, *NOMINAL_TIP])


def _lm(p, signs, samples, iters=200):
    lam = 1e-3
    r = _residuals(p, signs, samples)
    for _ in range(iters):
        J = np.empty((len(r), len(p)))
        for k in range(len(p)):
            dp = np.zeros(len(p))
            dp[k] = 1e-3 if k in (4, 5, 6) else 1e-6  # degrees vs meters/radians
            J[:, k] = (_residuals(p + dp, signs, samples) - r) / dp[k]
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -J.T @ r)
        r_new = _residuals(p + step, signs, samples)
        if r_new @ r_new < r @ r:
            p, r, lam = p + step, r_new, lam * 0.3
            if np.linalg.norm(step) < 1e-9:
                break
        else:
            lam *= 10
            if lam > 1e8:
                break
    return p


def fit(samples: list[tuple[dict, tuple[float, float]]]) -> tuple[ArmModel, np.ndarray]:
    """samples: (joint readings deg, touched target xy) -> (model, per-sample error mm)."""
    if len(samples) < 5:
        raise ValueError("Need at least 5 touches (9+ spread over the sheet is better).")
    sign_sets = [[1] * 5]
    sign_sets += [[s0, s1, s2, s3, 1] for s0 in (1, -1) for s1 in (1, -1) for s2 in (1, -1)
                  for s3 in (1, -1) if [s0, s1, s2, s3] != [1, 1, 1, 1]]
    best = None
    for signs in sign_sets:
        p = _lm(_kabsch_init(signs, samples), signs, samples)
        m = _unpack(p, signs)
        errs = np.array([np.linalg.norm(m.tip_position(q) - [x, y, 0]) for q, (x, y) in samples]) * 1000
        rms = float(np.sqrt(np.mean(errs**2)))
        if best is None or rms < best[0]:
            best = (rms, m, errs)
        if signs == [1] * 5 and rms < 8:  # URDF signs fit well: no need to search
            break
    rms, m, errs = best
    m.rms_mm = round(rms, 2)
    return m, errs

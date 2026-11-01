"""Named joint poses (degrees; gripper 0=closed..100=open), stored in poses.json."""

import json

from .config import POSES_FILE

Pose = dict[str, float]


def load_poses() -> dict[str, Pose]:
    if not POSES_FILE.exists():
        return {}
    return json.loads(POSES_FILE.read_text())


def save_pose(name: str, pose: Pose) -> None:
    poses = load_poses()
    poses[name] = {k: round(v, 2) for k, v in pose.items()}
    POSES_FILE.write_text(json.dumps(poses, indent=2) + "\n")


def require_pose(name: str, *aliases: str) -> Pose:
    """Return the first taught pose among `name` and its aliases."""
    poses = load_poses()
    for n in (name, *aliases):
        if n in poses:
            return poses[n]
    raise SystemExit(
        f"Pose '{name}' isn't taught yet. Run: python -m deskbot.teach  and save one called '{name}'."
    )

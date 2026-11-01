"""Play an emotion:  python -m deskbot.emote happy|sad   (Ctrl+C stops safely)"""

import sys

from .arm import Arm
from .emotions import EMOTIONS


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in EMOTIONS:
        raise SystemExit(f"usage: python -m deskbot.emote {{{'|'.join(EMOTIONS)}}}")
    with Arm() as arm:
        EMOTIONS[name](arm)


if __name__ == "__main__":
    main()

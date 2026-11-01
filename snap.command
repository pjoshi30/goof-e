#!/bin/zsh
# Double-click (or `open snap.command`) to grab a C920 frame from Terminal, which has its own camera permission.
cd "$(dirname "$0")"
~/miniforge3/envs/lerobot-mac/bin/python -m deskbot.snap 0

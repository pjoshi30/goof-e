#!/bin/zsh
# Double-click (or `open snap.command`) to grab a webcam frame from Terminal,
# which has its own camera permission. Saves it under snapshots/.
cd "$(dirname "$0")"
source ./scripts/find_python.sh
"$PY" -m deskbot.snap "${GOOFE_CAMERA_INDEX:-0}"

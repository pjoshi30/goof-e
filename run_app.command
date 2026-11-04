#!/bin/zsh
# Double-click (or `open run_app.command`) to start the voice web app from Terminal,
# which has camera access. Opens http://127.0.0.1:8765 when ready. Ctrl+C parks the arm.
cd "$(dirname "$0")"
source ./scripts/find_python.sh
"$PY" -m deskbot.webapp

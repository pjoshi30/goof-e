# so101-desk

Desk-organizing SO-101 arm: Claude plans, the camera locates objects, scripted moves do the picking. Voice commands and simple emotions.

Run everything from this folder with the `lerobot-mac` env:

```bash
conda activate lerobot-mac
```

| Command | What it does |
|---|---|
| `python -m deskbot.status` | Prints joint positions and limits. Never moves the arm. |
| `python -m deskbot.teach` | Follower mirrors the leader; type a name + Enter to save a pose. |
| `python -m deskbot.emote happy` / `sad` | Plays an emotion. |

| `python -m deskbot.targets` | Writes `calib/targets.pdf`, the calibration sheet. |
| `python -m deskbot.calib_camera` | Maps camera pixels to desk coordinates from the sheet's markers. |
| `python -m deskbot.calib_arm` | Touch the sheet's targets with the gripper tip; fits where the arm sits. |
| `python -m deskbot.touch_test` | Click the camera image; the arm touches that spot. |
| `python -m deskbot.hover 5 20` | Hold the tip 20 mm above target 5, to measure the real gap. |
| `python -m deskbot.perceive --background` | Save the empty desk (arm parked at rest); redo if lighting or the sheet changes. |
| `python -m deskbot.perceive` | Find objects; prints position, size, grip angle and writes `snapshots/objects.jpg`. |
| `open run_app.command` | **Voice web app** (Phase 5): starts in Terminal and opens http://127.0.0.1:8765. Hold to talk (or space bar), or type. Needs `ANTHROPIC_API_KEY` and `GEMINI_API_KEY` in `.env`. |
| `python -m deskbot.perceive --er2` | Same, but Gemini Robotics-ER 2 finds and names the objects (needs `GEMINI_API_KEY` in `.env`). |
| `python -m deskbot.pick_place` | Click an object, then where it should go; the arm moves it (run from Terminal). Uses ER 2; `--classic` for the empty-desk comparison only. |

Ctrl+C at any time freezes the arm, then parks it at `rest`.

Camera access: run camera commands from macOS Terminal (the Claude app's shell isn't allowed to use the camera).

## Calibration (Phase 2)

1. Print `calib/targets.pdf` at 100% and check the scale line measures 10.0 cm.
2. Tape it flat within the arm's reach and the camera view, ~15–30 cm from the base.
3. Snapshot (`python -m deskbot.snap 0`), then `python -m deskbot.calib_camera`. Check `calib/camera_check.jpg`.
4. `python -m deskbot.calib_arm`: touch each target (1–9, m0–m3) with the closed gripper tip, then `q`.
5. `python -m deskbot.touch_test`: click spots and see how close the tip lands.

Only the camera must stay put afterward; the sheet can be removed.

Teach these poses first: `rest`, `neutral`, `happy`, `sad` (see `deskbot/teach.py`).
Hardware settings live in `deskbot/config.py`.

## Findings / notes

- **2026-10-31, object detection.** Comparing the camera image with an empty-desk photo
  finds positions, sizes and grip angles precisely, but it can't tell touching objects apart
  (a marker, AirPods case and power bank merged into one blob), and white objects on the
  white sheet come out partial. Takeaway: classic object detection is still not solved by
  pairing a general-purpose LLM (e.g. Claude) with simple vision; not yet measured
  head-to-head. Next: try Gemini Robotics-ER 2 (robotics-tuned pointing/boxes) for
  finding and naming objects, and see how good it is. SAM was ruled out: too slow on a
  MacBook Air.

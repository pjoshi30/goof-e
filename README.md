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

Ctrl+C at any time freezes the arm, then parks it at `rest`.

Teach these poses first: `rest`, `neutral`, `happy_up`, `sad_bow` (see `deskbot/teach.py`).
Hardware settings live in `deskbot/config.py`.

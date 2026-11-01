"""Locate the arm on the desk by touching the printed targets with the gripper tip.

    python -m deskbot.calib_arm          # interactive (leader drives the follower)
    python -m deskbot.calib_arm --fit    # just refit from saved touches

Keep the gripper CLOSED (leader trigger released). Steer the follower so the very tip
of the jaws rests on a target's center, then type its name (1-9, m0-m3) + Enter.
Vary the wrist angle between touches (some steep, some tilted) — it helps the fit.

Commands: <target>  record a touch      undo  drop the last touch
          fit       fit and show errors  q     fit, save, and quit
"""

import json
import sys

import numpy as np

from .arm import Arm
from .kinematics import ARM_FILE, fit
from .targets import OUT_DIR, touch_targets
from .teleop import run_teleop

SAMPLES_FILE = OUT_DIR / "touch_samples.json"


def load_samples() -> list[dict]:
    return json.loads(SAMPLES_FILE.read_text()) if SAMPLES_FILE.exists() else []


def save_samples(samples: list[dict]) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SAMPLES_FILE.write_text(json.dumps(samples, indent=1) + "\n")


def fit_and_report(save: bool) -> None:
    raw = load_samples()
    targets = touch_targets()
    keep = list(range(len(raw)))
    try:
        # Drop up to 2 obvious outliers (wrong label, tip slipped), refitting each time.
        for _ in range(3):
            model, errs = fit([(raw[i]["q"], targets[raw[i]["target"]]) for i in keep])
            worst = int(np.argmax(errs))
            if len(keep) <= 8 or errs[worst] < max(15.0, 4 * float(np.median(errs))):
                break
            keep.pop(worst)
    except ValueError as e:
        print(f"  {e}")
        return
    err_of = dict(zip(keep, errs))
    print(f"  fit over {len(keep)} of {len(raw)} touches: RMS {model.rms_mm:.1f} mm")
    for i, s in enumerate(raw):
        note = f"{err_of[i]:5.1f} mm" if i in err_of else "  excluded (doesn't fit the rest — re-touch it?)"
        print(f"    #{i + 1:<2} {s['target']:>3}: {note}")
    offs = [round(float(o), 1) for o in model.offsets_deg[1:4]]
    print(f"  base at ({model.base_xyz[0]*100:.1f}, {model.base_xyz[1]*100:.1f}) cm, "
          f"joint offsets {offs} deg, signs {model.signs}")
    if save:
        model.save()
        print(f"  saved {ARM_FILE}")


def main() -> None:
    if "--fit" in sys.argv:
        fit_and_report(save=True)
        return

    samples = load_samples()
    if samples:
        print(f"{len(samples)} earlier touches loaded (type 'undo' to drop, or delete {SAMPLES_FILE}).")
    names = set(touch_targets())

    def on_command(cmd: str, arm: Arm) -> bool:
        if cmd == "q":
            fit_and_report(save=True)
            return False
        if cmd == "fit":
            fit_and_report(save=False)
        elif cmd == "undo":
            if samples:
                print(f"  dropped touch on {samples.pop()['target']}")
                save_samples(samples)
        elif cmd in names:
            q = arm.read()
            samples.append({"target": cmd, "q": q})
            save_samples(samples)
            print(f"  touch {len(samples)} recorded on {cmd} (gripper {q['gripper']:.0f})")
        else:
            print(f"  unknown command '{cmd}'. Targets: {', '.join(sorted(names))}")
        return True

    run_teleop(on_command, intro=__doc__)


if __name__ == "__main__":
    main()

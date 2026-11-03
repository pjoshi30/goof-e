"""Find and name desk objects with Gemini Robotics-ER 2, then outline each one precisely.

ER 2 is good at *what and roughly where*: it names every object and gives a centre point
and a bounding box, and it treats touching objects as separate things. It doesn't return
outlines, and a box doesn't say which way an object is turned. So we combine it with the
empty-desk comparison: ER 2's points seed a watershed split of the "something is here"
area, which cuts touching objects apart along the edges between them. Each piece then
goes through the usual geometry (desk position, size, grip angle).

Needs GEMINI_API_KEY (in the environment or in the project's .env file). Google rejects
unrestricted keys with 403: add an API restriction to the key in AI Studio.
"""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .camera import DeskMap
from .kinematics import ArmModel
from .perception import DeskObject, _work_area_test, change_mask, describe

MODEL = "gemini-robotics-er-2-preview"
THINKING_LEVEL = "medium"   # Google's suggested latency/quality balance for ER
SEND_WIDTH = 1280           # downscale before sending: faster, and plenty for pointing
ROOT = Path(__file__).resolve().parent.parent
LAST_RESPONSE = ROOT / "snapshots" / "er2_last.json"
SHORT_WAIT_S = 20          # rate limits shorter than this are retried once; longer ones fall back


class DetectorUnavailable(Exception):
    """ER 2 can't be used right now (quota, outage, bad key). The message says why, for people."""


_blocked_until = 0.0       # after a quota/billing error, don't ask again until then (time.time())
_blocked_why = ""


def _retry_after_s(err) -> float | None:
    """Google's suggested wait from a 429 (RetryInfo.retryDelay like '2787s'), if present."""
    try:
        for d in err.details.get("error", {}).get("details", []):
            if "RetryInfo" in d.get("@type", ""):
                return float(str(d["retryDelay"]).rstrip("s"))
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return None


def _quota_kind(err) -> str:
    text = str(getattr(err, "details", "")) + str(err)
    return "daily" if "PerDay" in text or "per day" in text.lower() else "rate"

PROMPT = """This is a top-down photo of a desk with a small robot arm on the right.
Find every separate loose object lying on the desk that a robot could pick up or that is part of
the desk's clutter.

Do NOT include: the robot arm or anything attached to it, its cables, the white paper sheet taped
to the desk, the black square markers printed on the sheet, the printed crosshairs, lines or text,
or the desk itself.

Objects that touch or overlap are still separate objects: list each one on its own.

For each object give a short label with its colour (e.g. "black marker", "white airpods case"),
a point at the centre of the object, and a tight bounding box.
Answer with JSON only, in this format:
[{"label": <label>, "point": [y, x], "box": [y_min, x_min, y_max, x_max]}, ...]
All coordinates are normalized to 0-1000."""


def _client():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit(f"No Gemini API key. Put GEMINI_API_KEY=... in {ROOT / '.env'} "
                         "(the key must have an API restriction set in AI Studio).")
    from google import genai
    return genai.Client()


def ask_er2(img: np.ndarray) -> tuple[list[dict], float]:
    """-> (objects with pixel 'point' (x, y) and 'box' (x0, y0, x1, y1), seconds taken)."""
    from google.genai import types

    h, w = img.shape[:2]
    small = cv2.resize(img, (SEND_WIDTH, round(h * SEND_WIDTH / w)))
    ok, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 90])
    from google.genai import errors

    global _blocked_until, _blocked_why
    if time.time() < _blocked_until:
        mins = max(1, round((_blocked_until - time.time()) / 60))
        raise DetectorUnavailable(f"{_blocked_why} (not retrying for about {mins} min)")

    client = _client()  # keep a reference: the client closes its connection when collected
    t0 = time.perf_counter()
    for attempt in (1, 2):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg"), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                ),
            )
            break
        except errors.APIError as e:
            if e.code == 429:
                wait = _retry_after_s(e)
                if attempt == 1 and wait is not None and wait <= SHORT_WAIT_S:
                    time.sleep(wait + 0.5)       # a brief per-minute limit: wait it out once
                    continue
                wait = wait if wait is not None else 60.0
                _blocked_until = time.time() + wait
                what = "daily free-tier quota" if _quota_kind(e) == "daily" else "rate limit"
                _blocked_why = f"Gemini ER 2 hit its {what}"
                raise DetectorUnavailable(f"{_blocked_why} (try again in about {max(1, round(wait / 60))} min)") from e
            if e.code in (500, 502, 503, 504) and attempt == 1:
                time.sleep(2)                    # transient server trouble: one retry
                continue
            if e.code == 402:                    # prepaid billing balance used up
                _blocked_until = time.time() + 600
                _blocked_why = "Gemini's prepaid credits are used up (top up the project's billing in AI Studio)"
                raise DetectorUnavailable(_blocked_why) from e
            if e.code in (401, 403):
                raise DetectorUnavailable("Gemini rejected the API key (it must have an API "
                                          "restriction set in AI Studio)") from e
            raise DetectorUnavailable(f"Gemini ER 2 returned an error ({e.code}: {e.message or e.status})") from e
    took = time.perf_counter() - t0
    text = response.text or "[]"
    LAST_RESPONSE.parent.mkdir(exist_ok=True)
    LAST_RESPONSE.write_text(text)

    items = json.loads(text)
    out = []
    for it in items if isinstance(items, list) else []:
        try:
            py, px = it["point"]
            y0, x0, y1, x1 = it["box"]
        except (KeyError, TypeError, ValueError):
            continue  # skip malformed entries rather than fail the whole look
        sx, sy = w / 1000, h / 1000
        out.append({"label": str(it.get("label", "")),
                    "point": (px * sx, py * sy),
                    "box": (min(x0, x1) * sx, min(y0, y1) * sy, max(x0, x1) * sx, max(y0, y1) * sy)})
    return out, took


def _box_outline(box) -> np.ndarray:
    x0, y0, x1, y1 = box
    return np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32)


PIECE_GAP_PX = 45   # extra pieces this close (~1.8 cm) to an object's main piece can belong to it
PIECE_MIN_PX = 300  # smaller stray pieces are speckle


def _inbox(shape, box, grow: bool = True) -> np.ndarray:
    """ER 2's box as a pixel mask, grown a little by default (boxes can be a little tight)."""
    x0, y0, x1, y1 = box
    mx, my = (0.15 * (x1 - x0) + 8, 0.15 * (y1 - y0) + 8) if grow else (0, 0)
    out = np.zeros(shape, bool)
    out[max(0, int(y0 - my)):int(y1 + my), max(0, int(x0 - mx)):int(x1 + mx)] = True
    return out


def split_by_points(img: np.ndarray, mask: np.ndarray, found: list[dict]) -> list[np.ndarray | None]:
    """Watershed the change mask with one seed per ER 2 object.
    -> outline per object, or None where the mask has (almost) nothing for it."""
    h, w = mask.shape
    n_comp, comp, stats, _ = cv2.connectedComponentsWithStats(mask)
    areas = stats[:, cv2.CC_STAT_AREA]

    # Which changed blob is each object's point in? (Snap to the nearest blob within ~6 mm.)
    blob_of = []
    for f in found:
        px, py = (int(round(v)) for v in f["point"])
        y0, y1, x0, x1 = max(0, py - 15), min(h, py + 16), max(0, px - 15), min(w, px + 16)
        win = comp[y0:y1, x0:x1]
        if 0 <= py < h and 0 <= px < w and comp[py, px]:
            blob_of.append(int(comp[py, px]))
        elif win.any():
            yy, xx = np.nonzero(win)
            i = np.argmin((yy + y0 - py) ** 2 + (xx + x0 - px) ** 2)
            blob_of.append(int(win[yy[i], xx[i]]))
        else:
            blob_of.append(0)

    # Label each blob's pixels by object: a blob with one object is all that object's;
    # a blob shared by several (touching objects) is split by watershed from their points.
    # The empty desk never competes inside a blob, so thin objects don't get nibbled away.
    labels = np.zeros((h, w), np.int32)
    for b in set(blob_of) - {0}:
        ks = [k for k, bb in enumerate(blob_of) if bb == b]
        inside = comp == b
        if len(ks) == 1:
            labels[inside] = ks[0] + 1
            continue
        bg_label = len(found) + 1
        markers = np.where(inside, 0, bg_label).astype(np.int32)  # outside the blob seeds "desk"
        for k in ks:
            px, py = (int(round(v)) for v in found[k]["point"])
            cv2.circle(markers, (px, py), 6, k + 1, -1)
        markers[~inside] = bg_label
        cv2.watershed(img, markers)
        markers[markers == bg_label] = 0
        # Blob pixels the desk claimed, plus watershed's boundary pixels (-1): nearest object
        lab = np.where(inside & (markers > 0), markers, 0)
        rest = inside & (lab == 0)
        if rest.any():
            dists = [cv2.distanceTransform((lab != k + 1).astype(np.uint8), cv2.DIST_L2, 3) for k in ks]
            nearest = np.array(ks)[np.argmin(np.stack(dists), axis=0)] + 1
            lab[rest] = nearest[rest]
        labels[inside] = lab[inside]

    # An object can show as several pieces (a marker's white label, a screwdriver's thin
    # shaft barely differ from the desk). Add pieces that nobody's point is on when they
    # sit inside this object's box and close to its main piece.
    unclaimed = [b for b in range(1, n_comp) if b not in blob_of and areas[b] >= PIECE_MIN_PX]
    for k, f in enumerate(found):
        own = labels == k + 1
        if not own.any():
            continue
        inbox = _inbox(mask.shape, f["box"], grow=False)   # strict: extra pieces need more proof
        near = cv2.dilate(own.astype(np.uint8), np.ones((2 * PIECE_GAP_PX + 1,) * 2, np.uint8)) > 0
        for b in unclaimed:
            piece = comp == b
            if (piece & near).any() and (piece & inbox).sum() > 0.8 * areas[b]:
                labels[piece & (labels == 0)] = k + 1

    outlines = []
    for k, f in enumerate(found):
        x0, y0, x1, y1 = f["box"]
        region = (labels == k + 1) & _inbox(mask.shape, f["box"])
        # A thin object lying diagonally fills only ~15% of its upright box, and sometimes
        # only its handle shows, so keep this low.
        box_area = max(1.0, (x1 - x0) * (y1 - y0))
        if region.sum() < max(300, 0.04 * box_area):
            outlines.append(None)  # e.g. white-on-white: the change mask barely sees it
            continue
        ys, xs = np.nonzero(region)
        outlines.append(cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)))
    return outlines


def find_objects_er2(img: np.ndarray, background: np.ndarray, desk: DeskMap,
                     model: ArmModel, limits: dict) -> tuple[list[DeskObject], dict]:
    """-> (objects, info) where info has ER 2's raw finds, timing and which outlines fell back to boxes."""
    found, took = ask_er2(img)
    mask = change_mask(img, background)
    outlines = split_by_points(img, mask, found)
    on_desk = _work_area_test()
    objects, used_box = [], []
    for f, c in zip(found, outlines):
        if c is None:
            c = _box_outline(f["box"])   # fall back to ER 2's box (orientation unknown)
            used_box.append(f["label"])
        obj = describe(c, desk, model, limits, on_desk, label=f["label"], check_area=False)
        if obj is not None:
            objects.append(obj)
    objects.sort(key=lambda o: o.reach_m)
    for i, o in enumerate(objects, 1):
        o.id = i
    return objects, {"er2": found, "seconds": took, "box_fallback": used_box}

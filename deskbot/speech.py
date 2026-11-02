"""Goof-E's speaking voice, via the OpenAI Realtime API.

OpenAI's text-to-speech models (gpt-4o-mini-tts, tts-1) shut down on 2027-01-06 and the
documented replacement is the Realtime API. We use it as a voice only: each line is an
out-of-band response with no conversation context, told to say the text exactly, in a
tone that fits (excited for "Yayy!", deflated for "Aww"). Claude still decides what to say.

Needs OPENAI_API_KEY (environment or the project's .env). Without it the web page falls
back to the browser's built-in voice.
"""

import asyncio
import base64
import io
import os
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("gpt-realtime-2.1-mini", "gpt-realtime-2.1")   # try the cheaper one first
VOICE = "marin"          # goofy persona; picked 2026-11-01 from samples (others tried: verse, ash, ballad, coral)
RATE = 24_000            # 16-bit mono PCM at 24 kHz

PERSONA = ("You are the voice of Goof-E, a small, clumsy, over-eager workshop robot arm, a lovable "
           "goofball sidekick like a cartoon robot. Bouncy, playful and a bit silly: big expressive "
           "pitch swings, little giggles and 'ooh!'s, comic timing, gets excited easily. Warm and "
           "friendly, never mean, never monotone.")
MOODS = {
    "happy": "Say it with bright, bouncy, genuine delight, like a cheer.",
    "sad": "Say it with a soft, deflated, disappointed sigh, drawn out a little.",
    "neutral": "",
}

_model_ok: str | None = None


def available() -> bool:
    _load_env()
    return bool(os.environ.get("OPENAI_API_KEY"))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


async def _speak_pcm(text: str, mood: str, model: str) -> bytes:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    pcm = bytearray()
    async with client.realtime.connect(model=model) as conn:
        await conn.session.update(session={
            "type": "realtime",
            "output_modalities": ["audio"],
            "audio": {"output": {"voice": VOICE, "format": {"type": "audio/pcm", "rate": RATE}}},
            "instructions": PERSONA,
        })
        await conn.response.create(response={
            "conversation": "none",
            "input": [],
            "output_modalities": ["audio"],
            "instructions": f"{MOODS.get(mood, '')} Say exactly the following, word for word, and "
                            f"nothing else:\n{text}",
        })
        async for event in conn:
            if event.type == "response.output_audio.delta":
                pcm += base64.b64decode(event.delta)
            elif event.type == "response.done":
                break
            elif event.type == "error":
                raise RuntimeError(getattr(event, "error", event))
    return bytes(pcm)


def speak_wav(text: str, mood: str = "neutral") -> bytes:
    """-> WAV bytes of Goof-E saying `text`. Raises if the API isn't reachable."""
    global _model_ok
    _load_env()
    last: Exception | None = None
    for model in ([_model_ok] if _model_ok else MODELS):
        try:
            pcm = asyncio.run(_speak_pcm(text, mood, model))
            if not pcm:
                raise RuntimeError("no audio returned")
            _model_ok = model
            break
        except Exception as e:  # e.g. model not available on this account: try the next
            last = e
    else:
        raise RuntimeError(f"Realtime voice failed: {last}")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


# ---- exclamations, generated once and kept, so they can play the instant a gesture starts ----

EXCLAIMS = {"happy": "Yayy!", "sad": "Aww…"}
CACHE_DIR = ROOT / "snapshots" / "voice_cache"


def hash_text(s: str) -> int:
    """Stable across runs (unlike hash()), so a changed persona or voice makes fresh clips."""
    import zlib
    return zlib.crc32(s.encode())


def exclaim_wav(mood: str) -> bytes | None:
    """Cached WAV for this mood's exclamation (generating it the first time), or None."""
    text = EXCLAIMS.get(mood)
    if text is None or not available():
        return None
    path = CACHE_DIR / f"{mood}-{VOICE}-{hash_text(PERSONA + MOODS.get(mood, '')):08x}.wav"
    if path.exists():
        return path.read_bytes()
    wav = speak_wav(text, mood)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav)
    return wav


def warm_exclaims() -> None:
    for mood in EXCLAIMS:
        try:
            exclaim_wav(mood)
        except Exception as e:  # no credits / offline: the page falls back to the browser voice
            print(f"couldn't prepare the '{mood}' voice clip: {e}")

"""Speech to text, on this Mac (MLX Whisper). Audio never leaves the machine.

The first call downloads the model (~1.6 GB) into the Hugging Face cache; after that a
few seconds of speech transcribes in about a second on Apple silicon.
"""

import threading

MODEL = "mlx-community/whisper-large-v3-turbo"
# Words Whisper might otherwise mishear in this setting.
VOCAB = "Desk robot commands: pick up, put, move, tidy, organize, the marker, pen, AirPods case, " \
        "USB stick, power bank, bracket, screwdriver, left, right, next to, happy, sad."

_lock = threading.Lock()


def transcribe(audio_path: str) -> str:
    import mlx_whisper
    with _lock:  # one at a time: MLX shares the GPU
        out = mlx_whisper.transcribe(audio_path, path_or_hf_repo=MODEL, language="en",
                                     initial_prompt=VOCAB, condition_on_previous_text=False)
    return str(out.get("text", "")).strip()


def warm_up() -> None:
    """Load (and if needed download) the model before the first real request."""
    import numpy as np
    import mlx_whisper
    with _lock:
        mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=MODEL, language="en")

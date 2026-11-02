"""Local web app: hold to talk, the robot listens, thinks (Claude), and acts.

    python -m deskbot.webapp       # run from Terminal (needs the camera); then open http://127.0.0.1:8765

Only listens on this Mac (127.0.0.1). Quitting (Ctrl+C) parks the arm.
"""

import asyncio
import json
import tempfile
import threading
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

from . import speech, voice
from .agent import Agent
from .desk import VIEW_FILE, Desk

HOST, PORT = "127.0.0.1", 8765
STATIC = Path(__file__).resolve().parent / "static"
MUSIC_DIR = Path(__file__).resolve().parent.parent / "music"
MUSIC_TYPES = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg"}

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Connecting to the arm…")
    desk = Desk()
    state["desk"], state["agent"], state["busy"] = desk, Agent(desk), False
    state["listeners"] = set()   # every open page gets arm-motion events (for the music)
    desk.arm.on_motion = lambda moving: [f("motion", {"on": moving}) for f in list(state["listeners"])]
    threading.Thread(target=voice.warm_up, daemon=True).start()
    threading.Thread(target=speech.warm_exclaims, daemon=True).start()
    print(f"Ready: http://{HOST}:{PORT}")
    yield
    print("Shutting down…")
    desk.stop()
    desk.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/view.jpg")
def view():
    if not VIEW_FILE.exists():
        return JSONResponse({"error": "no view yet"}, status_code=404)
    return FileResponse(VIEW_FILE, headers={"Cache-Control": "no-store"})


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "speech.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(await audio.read())
        f.flush()
        text = await asyncio.to_thread(voice.transcribe, f.name)
    return {"text": text}


def _music_file() -> Path | None:
    files = sorted(f for f in MUSIC_DIR.glob("*") if f.suffix.lower() in MUSIC_TYPES) if MUSIC_DIR.exists() else []
    return files[0] if files else None


@app.get("/api/music")
def music_info():
    f = _music_file()
    return {"url": f"/music/{f.name}" if f else None, "name": f.stem if f else None}


@app.get("/music/{name}")
def music_file(name: str):
    f = _music_file()
    if f is None or name != f.name:   # only ever serve the one chosen file
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(f, media_type=MUSIC_TYPES[f.suffix.lower()])


@app.get("/api/voice")
def voice_status():
    return {"natural": speech.available()}


@app.get("/api/exclaim/{mood}.wav")
async def exclaim(mood: str):
    try:
        wav = await asyncio.to_thread(speech.exclaim_wav, mood)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    if wav is None:
        return JSONResponse({"error": "not available"}, status_code=404)
    return Response(wav, media_type="audio/wav")


@app.post("/api/speak")
async def speak(body: dict):
    text = str(body.get("text", "")).strip()[:600]
    if not text or not speech.available():
        return JSONResponse({"error": "no natural voice configured"}, status_code=503)
    try:
        wav = await asyncio.to_thread(speech.speak_wav, text, str(body.get("mood", "neutral")))
    except Exception as e:
        print(f"voice error: {e}")
        return JSONResponse({"error": str(e)}, status_code=502)
    return Response(wav, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@app.websocket("/ws")
async def ws(socket: WebSocket):
    await socket.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, data: dict) -> None:   # called from the agent's worker thread
        loop.call_soon_threadsafe(outbox.put_nowait, {"type": kind, **data})

    async def pump():
        while True:
            await socket.send_text(json.dumps(await outbox.get()))

    async def run(text: str):
        state["busy"] = True
        emit("busy", {"on": True})
        try:
            reply = await asyncio.to_thread(state["agent"].handle, text, emit)
            emit("reply", {"text": reply})
        except Exception as e:  # keep the app alive whatever happens
            emit("reply", {"text": f"Something went wrong: {type(e).__name__}: {e}"})
        finally:
            state["busy"] = False
            emit("busy", {"on": False})

    pumper = asyncio.create_task(pump())
    state["listeners"].add(emit)
    try:
        while True:
            msg = json.loads(await socket.receive_text())
            kind = msg.get("type")
            if kind == "stop":
                state["desk"].stop()
                emit("status", {"text": "Stopping…"})
            elif kind == "reset":
                if state["busy"]:
                    emit("status", {"text": "Busy: stop first, then start a new conversation."})
                else:
                    state["agent"].reset()
                    emit("status", {"text": "New conversation."})
            elif kind == "command" and msg.get("text", "").strip():
                if state["busy"]:
                    emit("status", {"text": "Still working on the last request (press Stop to interrupt)."})
                else:
                    asyncio.create_task(run(msg["text"].strip()))
    except WebSocketDisconnect:
        pass
    finally:
        state["listeners"].discard(emit)
        pumper.cancel()


def main() -> None:
    import uvicorn
    threading.Timer(2.5, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

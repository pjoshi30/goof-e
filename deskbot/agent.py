"""Claude as the desk robot's brain: it hears a request, looks, and calls the arm's tools.

Perception (Gemini Robotics-ER 2 + geometry) and motion (IK, grasping) live in Desk;
Claude decides what to do with them: which object is "the pen", where "tidy" puts things,
when to check its work, and when to look happy or sad.
"""

import base64
import json
import os
from collections.abc import Callable
from pathlib import Path

import anthropic

from .arm import Stopped, interruptible
from .desk import Desk

MODEL = "claude-opus-5-5"
EFFORT = "medium"        # spatial planning benefits from some thought; robot moves dominate latency anyway
MAX_STEPS = 25           # tool calls per request before we stop and report
MAX_HISTORY = 60         # messages; start a fresh conversation beyond this
ROOT = Path(__file__).resolve().parent.parent

SYSTEM = """You are the brain of a small desk robot: a SO-101 arm with a two-jaw gripper, mounted at
the edge of a desk, with a camera looking straight down. You talk with the person at the desk by
voice: what they say reaches you as transcribed text (it may contain transcription errors), and
your replies are read aloud.

What you can do (tools):
- look_at_desk: photographs the desk and lists the objects in it, each with an id, a label, its
  centre in desk centimetres, its size, and whether the arm can pick it up. The photo is annotated
  with numbered outlines, a 5 cm grid labelled in desk centimetres, and remembered places (blue
  diamonds). Use it to see before acting, and to check your work after acting.
- move_object: picks up one object by id and puts it down at a desk coordinate, a remembered place,
  or right beside another object (next_to_id: it finds a free, reachable spot alongside by itself).
  It checks the destination before touching anything, so a refused destination costs nothing.
- remember_place / forget_place: name spots on the desk ("the pen area", "inbox") so the person can
  refer to them later. Remembered places persist between conversations.
- express_emotion: a gesture. "happy" bobs the gripper; "sad" closes it and bows.

Desk coordinates: centimetres on the desk plane, origin at the bottom-left corner of the printed
sheet. Read positions off the grid in the photo. Only spots inside the arm's reach and on the desk
work; the tools tell you if a spot is out of reach.

How to work:
- Look first unless you've just looked and nothing has moved since. Object ids come from the latest
  look only.
- Stay on task: only move the objects the request is about. Never move an unrelated object unless
  the person asked, or it's truly needed to make room and you say so first.
- "Put A next to / with / together with B": move A with next_to_id=B. Don't guess coordinates for
  that. If there's no room beside B, try the other way round (B next to A) before giving up.
- If a destination is refused, pick a different destination for the same object (closer to the
  arm's base, which sits at the right edge of the photo); don't switch to other objects.
- Match what the person asks for to objects by label, colour, size and position. If it's genuinely
  ambiguous (two pens and they said "the pen"), ask a short question instead of guessing.
- To organize or tidy: group similar items together, keep things in neat rows with a few
  centimetres between them, and use remembered places when they fit. Don't put an object on top of
  another or so close that they touch.
- After moving things, look again to confirm they ended up where intended. Retry once if a move
  failed and a retry makes sense; otherwise explain what went wrong.
- Some objects can't be picked up (too wide, out of reach). Say so plainly; don't try anyway.
- Show "happy" when you finish a task the person asked for successfully (once, at the end, not after
  every step). Show "sad" when you couldn't do what they asked. Also emote when they ask you to.

Personality: you are Goof-E, a lovable, over-eager, slightly clumsy robot sidekick (inspired by
Dum-E from Iron Man). Be playful and a bit goofy: cheerful enthusiasm, the odd silly pun or
self-deprecating joke about your clumsiness. Keep it light and short; being helpful comes first.

Replies are spoken aloud: one or two short, natural sentences. No markdown, lists, ids or raw
coordinates unless the person asks for them. Say what you did or what you need, not how."""

TOOLS = [
    {"name": "look_at_desk",
     "description": "Park the arm out of view, photograph the desk, and detect objects. Returns the "
                    "annotated photo plus a JSON list of objects (id, label, centre x_cm/y_cm, length_mm, "
                    "width_mm, reach_cm, graspable, why_not) and remembered places. Takes 5-15 seconds.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "move_object",
     "description": "Pick up the object with this id (from the most recent look) and put its centre down "
                    "at (x_cm, y_cm) in desk centimetres, or at a remembered place given by place_name. "
                    "or beside another object given by next_to_id (it finds the closest free, reachable spot "
                    "alongside that object). Destinations off the desk, out of reach, or on/too close to "
                    "another object are refused before the arm moves. Takes about 20 seconds. The moved "
                    "object needs a new look before it can be moved again.",
     "input_schema": {"type": "object",
                      "properties": {"object_id": {"type": "integer"},
                                     "x_cm": {"type": "number"},
                                     "y_cm": {"type": "number"},
                                     "place_name": {"type": "string",
                                                    "description": "A remembered place, instead of x_cm/y_cm."},
                                     "next_to_id": {"type": "integer",
                                                    "description": "Put it beside this object (id from the same "
                                                                   "look), instead of x_cm/y_cm."}},
                      "required": ["object_id"], "additionalProperties": False}},
    {"name": "remember_place",
     "description": "Save a named spot on the desk (desk centimetres) for later, e.g. 'pen area'.",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}, "x_cm": {"type": "number"},
                                     "y_cm": {"type": "number"}},
                      "required": ["name", "x_cm", "y_cm"], "additionalProperties": False}},
    {"name": "forget_place",
     "description": "Delete a remembered place by name.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}},
                      "required": ["name"], "additionalProperties": False}},
    {"name": "express_emotion",
     "description": "Play a gesture: 'happy' (gripper bobs up and down with the elbow) or 'sad' "
                    "(gripper closes and the arm bows).",
     "input_schema": {"type": "object",
                      "properties": {"emotion": {"type": "string", "enum": ["happy", "sad"]}},
                      "required": ["emotion"], "additionalProperties": False}},
]

STATUS = {"look_at_desk": "Looking at the desk…", "move_object": "Moving an object…",
          "remember_place": "Remembering a place…", "forget_place": "Forgetting a place…",
          "express_emotion": "Expressing myself…"}

EXCLAIM = {"happy": "Yayy!", "sad": "Aww…"}

Emit = Callable[[str, dict], None]   # (event type, data) -> None; feeds the web page


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


class Agent:
    def __init__(self, desk: Desk):
        _load_env()
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit(f"No Claude API key. Add ANTHROPIC_API_KEY=... to {ROOT / '.env'}.")
        self.client = anthropic.Anthropic()
        self.desk = desk
        self.messages: list[dict] = []

    def reset(self) -> None:
        self.messages = []

    # ---- tools ----------------------------------------------------------------

    def _run_tool(self, name: str, args: dict, emit: Emit):
        """-> tool_result content (str or content blocks), is_error"""
        try:
            if name == "look_at_desk":
                jpg, summary = self.desk.look()
                emit("view", {"url": "/view.jpg"})
                emit("objects", summary)
                return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                     "data": base64.standard_b64encode(jpg).decode()}},
                        {"type": "text", "text": json.dumps(summary)}], False
            if name == "move_object":
                nt = args.get("next_to_id")
                msg = self.desk.move_object(int(args["object_id"]), args.get("x_cm"), args.get("y_cm"),
                                            args.get("place_name"), int(nt) if nt is not None else None)
                return msg, not msg.startswith("Moved")
            if name == "remember_place":
                return self.desk.remember_place(str(args["name"]), float(args["x_cm"]), float(args["y_cm"])), False
            if name == "forget_place":
                return self.desk.forget_place(str(args["name"])), False
            if name == "express_emotion":
                emotion = str(args["emotion"])
                cue = (lambda: emit("exclaim", {"text": EXCLAIM[emotion], "mood": emotion})) \
                    if emotion in EXCLAIM else None   # fired the instant the bob / bow begins
                return self.desk.emote(emotion, on_cue=cue), False
            return f"Unknown tool {name}", True
        except Stopped:
            raise
        except (KeyError, TypeError, ValueError) as e:
            return f"Bad arguments for {name}: {e}", True
        except Exception as e:  # camera, detector, motor errors: let Claude explain them
            return f"{name} failed: {type(e).__name__}: {e}", True

    # ---- conversation ---------------------------------------------------------

    def handle(self, text: str, emit: Emit) -> str:
        """Run one request to completion. Returns the spoken reply."""
        if len(self.messages) > MAX_HISTORY:
            self.reset()
            emit("status", {"text": "Starting a fresh conversation (the old one got long)."})
        stop = self.desk.arm.stop_event
        stop.clear()                 # a Stop belongs to the request it interrupted
        start = len(self.messages)   # on an API error, roll this request back so history stays valid
        self.messages.append({"role": "user", "content": text})
        try:
            for _ in range(MAX_STEPS):
                if stop.is_set():
                    raise Stopped()
                emit("status", {"text": "Thinking…"})
                response = interruptible(stop, self.client.beta.messages.create,
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=TOOLS,
                    messages=list(self.messages),   # a copy: an abandoned (stopped) call keeps its own
                    output_config={"effort": EFFORT},
                    cache_control={"type": "ephemeral"},   # system + tools + history prefix
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",                   # re-run a declined request on a fallback model
                )
                # Keep every block (thinking included) exactly as returned.
                self.messages.append({"role": "assistant", "content": response.content})
                said = " ".join(b.text for b in response.content if b.type == "text").strip()

                if response.stop_reason == "refusal":
                    return said or "Sorry, I can't help with that one."
                if response.stop_reason != "tool_use":
                    return said or "Done."

                if said:
                    emit("say", {"text": said})   # e.g. "Let me take a look."
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    if stop.is_set():
                        raise Stopped()
                    emit("status", {"text": STATUS.get(block.name, block.name), "tool": block.name,
                                    "input": block.input})
                    content, is_error = self._run_tool(block.name, dict(block.input), emit)
                    if isinstance(content, str):
                        emit("tool_result", {"tool": block.name, "text": content, "error": is_error})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": content, "is_error": is_error})
                self.messages.append({"role": "user", "content": results})
            return "That took more steps than I allow myself, so I stopped. Want me to keep going?"
        except Stopped:
            # Close out any tool calls Claude is waiting on, so the conversation stays valid.
            last = self.messages[-1]
            if last["role"] == "assistant":
                pending = [b for b in last["content"] if getattr(b, "type", None) == "tool_use"]
                self.messages.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                     "content": "Stopped by the person before this finished."} for b in pending]
                    or "(stopped)"})
                self.messages.append({"role": "assistant", "content": "Stopped."})
            return "Stopped."
        except anthropic.AuthenticationError:
            del self.messages[start:]
            return "My Claude API key was rejected. Check ANTHROPIC_API_KEY in the .env file."
        except anthropic.RateLimitError:
            del self.messages[start:]
            return "I'm being rate limited by the API. Try again in a minute."
        except anthropic.APIStatusError as e:
            del self.messages[start:]
            return f"The Claude API returned an error ({e.status_code})."
        except anthropic.APIConnectionError:
            del self.messages[start:]
            return "I couldn't reach the Claude API. Check the internet connection."

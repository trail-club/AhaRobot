"""Records shown by the robot monitor and their HTML (ROS free).

Modules publish JSON (std_msgs/String) on three topics:

- STATUS_TOPIC: ``{"source", "state", "detail", "stamp"}``, the current state
  of a source (e.g. place_runner "deciding", place_server "approaching");
  each publisher's latest one is latched.
- CHAT_TOPIC: ``{"source", "id", "title", "messages", "notes", "done"}``, an
  LLM conversation so far: ``messages`` are chat completions messages
  (system, user, assistant with ``tool_calls``, tool), ``notes`` maps a
  message index (str) to notes ``{"text", "level"}`` shown after it (e.g.
  what the robot did with that answer). A record replaces the earlier one
  with the same source and id.
- EVENT_TOPIC: ``{"source", "level", "text", "stamp"}``, one log line;
  ``level`` is "info", "warn" or "error".

``stamp`` is wall-clock seconds (time.time()).
"""

from __future__ import annotations

import html
import json
import time

STATUS_TOPIC = "/aha/ui/status"
CHAT_TOPIC = "/aha/ui/chat"
EVENT_TOPIC = "/aha/ui/events"
LEVELS = ("info", "warn", "error")
LEVEL_COLOR = {
    "info": "#202020",
    "warn": "#9a6700",
    "error": "#b3261e",
    "ok": "#1b7f2a",
}
ROLE_STYLE = {
    # role: (title, background, align)
    "system": ("system", "#eeeeee", "left"),
    "user": ("user", "#e3f0ff", "left"),
    "assistant": ("assistant", "#e6f7e1", "right"),
    "tool": ("tool result", "#fff4dc", "left"),
}


def status(source: str, state: str, detail: str = "", stamp: float | None = None):
    return {
        "source": source,
        "state": state,
        "detail": detail,
        "stamp": time.time() if stamp is None else stamp,
    }


def event(source: str, text: str, level: str = "info", stamp: float | None = None):
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return {
        "source": source,
        "level": level,
        "text": text,
        "stamp": time.time() if stamp is None else stamp,
    }


def chat(
    source: str,
    chat_id: str,
    title: str,
    messages,
    notes: dict | None = None,
    done: bool = False,
):
    return {
        "source": source,
        "id": chat_id,
        "title": title,
        "messages": list(messages or []),
        "notes": {str(k): list(v) for k, v in (notes or {}).items()},
        "done": done,
    }


def note(text: str, level: str = "info"):
    """A note after a chat message; ``level`` also "ok"."""
    return {"text": text, "level": level}


def _text(value) -> str:
    return html.escape(str(value)).replace("\n", "<br>")


def _tool_call(call) -> str:
    function = call.get("function", {})
    name = function.get("name", "?")
    raw = function.get("arguments", "")
    try:
        arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (ValueError, TypeError):
        return f"<b>{_text(name)}</b>({_text(raw)})"
    reason = arguments.pop("reason", None)
    fields = ", ".join(f"{_text(k)}: <b>{_text(v)}</b>" for k, v in arguments.items())
    out = f"<b>{_text(name)}</b>({fields})"
    if reason:
        out += f"<br><i>{_text(reason)}</i>"
    return out


def _bubble(role: str, body: str) -> str:
    title, color, align = ROLE_STYLE.get(role, (role, "#ffffff", "left"))
    return (
        f'<table width="88%" align="{align}" cellpadding="6" '
        f'style="margin-bottom:6px; background-color:{color};">'
        f"<tr><td><small><b>{_text(title)}</b></small><br>{body}</td></tr></table>"
        '<br clear="all">'
    )


def _note(item) -> str:
    color = LEVEL_COLOR.get(item.get("level", "info"), LEVEL_COLOR["info"])
    return f'<p style="color:{color};"><b>→ {_text(item.get("text", ""))}</b></p>'


def chat_html(record: dict, show_system: bool = False) -> str:
    """HTML of a chat record: a bubble per message, its notes after it."""
    state = "" if record.get("done") else " …"
    parts = [
        f"<h3>{_text(record.get('title') or record.get('id', ''))}"
        f"<small> ({_text(record.get('source', '?'))}){state}</small></h3>"
    ]
    notes = record.get("notes", {})
    for k, message in enumerate(record.get("messages", [])):
        role = message.get("role", "")
        if role != "system" or show_system:
            body = _text(message.get("content") or "")
            calls = message.get("tool_calls") or []
            if calls:
                body = "<br>".join(
                    filter(None, [body, *(_tool_call(c) for c in calls)])
                )
            parts.append(_bubble(role, body))
        parts += [_note(n) for n in notes.get(str(k), [])]
    return "".join(parts)


def event_html(record: dict) -> str:
    color = LEVEL_COLOR.get(record.get("level", "info"), LEVEL_COLOR["info"])
    clock = time.strftime("%H:%M:%S", time.localtime(record.get("stamp", 0.0)))
    return (
        f'<span style="color:{color};">{clock} [{_text(record.get("source", "?"))}] '
        f"{_text(record.get('text', ''))}</span>"
    )


def age_text(stamp: float, now: float | None = None) -> str:
    seconds = max(0.0, (time.time() if now is None else now) - stamp)
    if seconds < 60:
        return f"{seconds:.0f} s"
    return f"{seconds / 60:.0f} min"

import json

import pytest

from aha_test_ui import records


def test_chat_html_puts_notes_after_their_message():
    messages = [
        {"role": "system", "content": "You decide <where>."},
        {"role": "user", "content": "Held object: apple"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "choose_place",
                        "arguments": json.dumps(
                            {"reason": "fruit", "level": 4, "next_to": "O1"}
                        ),
                    }
                }
            ],
        },
    ]
    record = records.chat(
        "place_runner",
        "apple-1",
        "apple (C)",
        messages,
        {2: [records.note("place: OK", "ok")]},
        done=True,
    )
    page = records.chat_html(record)
    assert "You decide" not in page  # system hidden by default
    assert "&lt;where&gt;" in records.chat_html(record, show_system=True)
    assert page.index("choose_place") < page.index("place: OK")
    assert "<i>fruit</i>" in page and "next_to: <b>O1</b>" in page
    assert "…" not in page and "apple (C)" in page


def test_status_and_event_records():
    s = records.status("place_server", "approaching", "candidate 1", stamp=10.0)
    assert s == {
        "source": "place_server",
        "state": "approaching",
        "detail": "candidate 1",
        "stamp": 10.0,
    }
    e = records.event("place_runner", "placed", "info", stamp=0.0)
    assert "[place_runner] placed" in records.event_html(e)
    with pytest.raises(ValueError):
        records.event("x", "y", "fatal")
    assert records.age_text(10.0, now=15.0) == "5 s"
    assert records.age_text(0.0, now=180.0) == "3 min"

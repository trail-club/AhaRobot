"""Helpers shared by the workers that speak JSON lines over an SSH stdio channel.

Standard library only, so that every worker image can import this module.
"""

from __future__ import annotations

import json
import os
import re
import sys


def safe_error(error: Exception) -> str:
    message = str(error)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[redacted]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message)


def send_json(stream, response: dict) -> None:
    stream.write(json.dumps(response, allow_nan=False) + "\n")
    stream.flush()


def protocol_stream():
    """Return the original stdout and point fd 1 at stderr.

    redirect_stdout only covers Python writes; native libraries can still write
    to fd 1 directly, which would corrupt the JSON lines read by the client.
    """
    sys.stdout.flush()
    stream = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    return stream

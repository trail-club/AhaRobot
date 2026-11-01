import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from aha_test_orchestration.llm import (
    InvalidAnswer,
    LlmCall,
    LlmClient,
    LlmError,
    ask_tool,
    tool_arguments,
    tool_results,
)

TOOL = {
    "type": "function",
    "function": {"name": "choose_place", "parameters": {"type": "object"}},
}


def call_message(arguments, call_id="c1"):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "choose_place", "arguments": arguments},
            }
        ],
    }


class ScriptedClient:
    """Mock LLM: returns the scripted messages in order, records the requests."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def chat(self, messages, tools=None, tool_choice=None):
        self.requests.append((list(messages), tools, tool_choice))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, LlmCall(0.5, 100, 20, "tool_calls")


def validate_level(arguments):
    if arguments.get("level") not in (3, 4):
        raise InvalidAnswer(f"unknown level {arguments.get('level')}")
    return arguments["level"]


def test_tool_results_answer_each_call_or_add_a_user_message():
    two = call_message('{"level": 4}')
    two["tool_calls"].append(dict(two["tool_calls"][0], id="c2"))
    results = tool_results([two], "failed")
    assert [(m["role"], m["tool_call_id"], m["content"]) for m in results] == [
        ("tool", "c1", "failed"),
        ("tool", "c2", "failed"),
    ]
    assert tool_results([{"role": "assistant", "content": "x"}], "failed") == [
        {"role": "user", "content": "failed"}
    ]
    assert tool_results([], "failed") == [{"role": "user", "content": "failed"}]


def test_tool_arguments():
    assert tool_arguments(call_message('{"level": 3}'), "choose_place") == {"level": 3}
    with pytest.raises(InvalidAnswer, match="not valid JSON"):
        tool_arguments(call_message("{level"), "choose_place")
    with pytest.raises(InvalidAnswer, match="no call"):
        tool_arguments({"role": "assistant", "content": "level 3"}, "choose_place")


def test_valid_first_answer():
    client = ScriptedClient([call_message('{"level": 4}')])
    answer = ask_tool(client, [{"role": "user", "content": "q"}], TOOL, validate_level)
    assert answer.value == 4 and answer.error == ""
    assert answer.prompt_tokens == 100 and answer.completion_tokens == 20
    _, tools, choice = client.requests[0]
    assert tools == [TOOL]
    assert choice == {"type": "function", "function": {"name": "choose_place"}}


def test_retry_with_error_then_valid():
    client = ScriptedClient(
        [call_message('{"level": 9}'), call_message('{"level": 3}', "c2")]
    )
    answer = ask_tool(client, [{"role": "user", "content": "q"}], TOOL, validate_level)
    assert answer.value == 3 and len(answer.calls) == 2
    assert answer.calls[0].error == "unknown level 9"
    retry_messages = client.requests[1][0]
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["tool_call_id"] == "c1"
    assert "unknown level 9" in retry_messages[-1]["content"]
    assert answer.latency_s == 1.0 and answer.prompt_tokens == 200


def test_fails_after_one_retry():
    client = ScriptedClient(
        [call_message('{"level": 9}'), {"role": "assistant", "content": "level 3"}]
    )
    answer = ask_tool(client, [{"role": "user", "content": "q"}], TOOL, validate_level)
    assert answer.value is None and "no call" in answer.error
    assert len(client.requests) == 2


def test_server_error_is_not_retried():
    client = ScriptedClient([LlmError("request failed: refused")])
    answer = ask_tool(client, [{"role": "user", "content": "q"}], TOOL, validate_level)
    assert answer.value is None and "refused" in answer.error and not answer.calls


class Handler(BaseHTTPRequestHandler):
    bodies = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.bodies.append((self.path, body))
        reply = {
            "choices": [
                {"message": call_message('{"level": 3}'), "finish_reason": "tool_calls"}
            ],
            "usage": {"prompt_tokens": 321, "completion_tokens": 12},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_client_request_and_usage():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = LlmClient(f"http://127.0.0.1:{server.server_port}/v1", "m", timeout=5)
        message, call = client.chat(
            [{"role": "user", "content": "q"}], [TOOL], {"type": "function"}
        )
    finally:
        server.shutdown()
    path, body = Handler.bodies[-1]
    assert path == "/v1/chat/completions"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.0 and body["max_tokens"] == 1000
    assert body["tools"] == [TOOL]
    assert call.prompt_tokens == 321 and call.completion_tokens == 12
    assert tool_arguments(message, "choose_place") == {"level": 3}


def test_client_connection_error_is_retried(monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(1)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    client = LlmClient("http://127.0.0.1:9/v1", "m", connect_retries=2, retry_wait=0.0)
    with pytest.raises(LlmError, match="refused"):
        client.chat([{"role": "user", "content": "q"}])
    assert len(attempts) == 3


def test_client_timeout_is_not_retried(monkeypatch):
    attempts = []

    def slow(*args, **kwargs):
        attempts.append(1)
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", slow)
    client = LlmClient("http://127.0.0.1:9/v1", "m", retry_wait=0.0)
    with pytest.raises(LlmError, match="timed out"):
        client.chat([{"role": "user", "content": "q"}])
    assert len(attempts) == 1

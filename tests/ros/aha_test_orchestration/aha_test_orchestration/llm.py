"""Small client of an OpenAI-compatible chat completions server (stdlib only).

``LlmClient.chat`` sends one request (repeated up to ``connect_retries``
times on connection errors, not on time-outs or HTTP errors) and returns the
assistant message with the measured latency and the token usage. ``ask_tool`` forces a function call,
validates its arguments and retries once with the validation error appended
to the conversation; ``tool_results`` answers the last call (e.g. with an
execution failure) before asking again.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_URL = "http://10.99.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b"


class LlmError(RuntimeError):
    """The server failed or answered without a usable function call."""


class InvalidAnswer(ValueError):
    """The function call arguments do not name a valid place."""


@dataclass
class LlmCall:
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    error: str = ""


@dataclass
class ToolAnswer:
    """Result of ``ask_tool``: the validated value, or None with the error."""

    value: object
    arguments: dict | None
    calls: list = field(default_factory=list)  # LlmCall per request
    messages: list = field(default_factory=list)  # full conversation
    error: str = ""

    @property
    def latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)


class LlmClient:
    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        thinking: bool = False,
        connect_retries: int = 2,
        retry_wait: float = 2.0,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.thinking = thinking
        self.connect_retries = connect_retries
        self.retry_wait = retry_wait

    def chat(self, messages, tools=None, tool_choice=None) -> tuple[dict, LlmCall]:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        if tools:
            body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(self.connect_retries + 1):
            start = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                break
            except urllib.error.HTTPError as e:
                raise LlmError(f"request failed: HTTP {e.code} {e.reason}") from e
            except urllib.error.URLError as e:
                # Connection errors (network down): retry; time-outs are not.
                timed_out = isinstance(e.reason, TimeoutError)
                if timed_out or attempt == self.connect_retries:
                    raise LlmError(f"request failed: {e}") from e
                time.sleep(self.retry_wait)
            except (TimeoutError, json.JSONDecodeError) as e:
                raise LlmError(f"request failed: {e}") from e
        latency = time.monotonic() - start
        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LlmError(f"unexpected response: {str(data)[:200]}") from e
        usage = data.get("usage") or {}
        return message, LlmCall(
            latency,
            int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)),
            str(choice.get("finish_reason", "")),
        )


def tool_arguments(message: dict, name: str) -> dict:
    """Arguments of the call of function ``name`` in an assistant message."""
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") != name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return arguments
        try:
            parsed = json.loads(arguments or "")
        except json.JSONDecodeError as e:
            raise InvalidAnswer(f"the arguments are not valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise InvalidAnswer("the arguments must be a JSON object")
        return parsed
    raise InvalidAnswer(f"no call of the function {name}")


def tool_results(messages, text: str) -> list:
    """Messages answering the last assistant message's function calls with
    ``text`` (a user message if it has none)."""
    last = messages[-1] if messages else {}
    calls = last.get("tool_calls") if last.get("role") == "assistant" else None
    if not calls:
        return [{"role": "user", "content": text}]
    return [
        {"role": "tool", "tool_call_id": c.get("id", ""), "content": text}
        for c in calls
    ]


def ask_tool(client, messages, tool: dict, validate, retries: int = 1) -> ToolAnswer:
    """Force a call of ``tool`` and validate it, retrying with the error.

    validate(arguments) returns the value or raises InvalidAnswer. Server
    errors are not retried."""
    name = tool["function"]["name"]
    messages = list(messages)
    answer = ToolAnswer(None, None, messages=messages)
    for attempt in range(retries + 1):
        try:
            message, call = client.chat(
                messages,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": name}},
            )
        except LlmError as e:
            answer.error = str(e)
            return answer
        answer.calls.append(call)
        assistant = {"role": "assistant", "content": message.get("content") or ""}
        if message.get("tool_calls"):
            assistant["tool_calls"] = message["tool_calls"]
        messages.append(assistant)
        try:
            arguments = tool_arguments(message, name)
            answer.arguments = arguments
            answer.value = validate(arguments)
            answer.error = ""
            return answer
        except InvalidAnswer as e:
            answer.error = str(e)
            call.error = str(e)
            if attempt == retries:
                break
            text = f"Invalid answer: {e}. Call {name} again with a valid answer."
            messages.extend(tool_results(messages, text))
    return answer

"""Chat with an OpenAI-compatible LLM server (the DGX Qwen by default), without ROS.

The server is TensorFold (tools/local-llm): it emits `reasoning_content`
unless thinking is turned off (chat_template_kwargs.enable_thinking), counts it
in `max_tokens`, and constrains `content` to a JSON schema given as
response_format {"type": "json_schema"}. `parameters(node)` reads the URL and
model from the runner's ROS parameters `llm_url` / `llm_model`.
"""

from dataclasses import dataclass
import json
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://10.99.0.1:8080/v1"
DEFAULT_MODEL = "qwen3.8-27b"


class LlmError(RuntimeError):
    """The server could not be reached or gave no usable answer."""


@dataclass
class Reply:
    content: str
    seconds: float


def parameters(node):
    """(url, model) from the node's ROS parameters, declared on first use."""
    values = []
    for name, default in (("llm_url", DEFAULT_URL), ("llm_model", DEFAULT_MODEL)):
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
        values.append(node.get_parameter(name).value or default)
    return tuple(values)


class Chat:
    """Chat completions of one server and model."""

    def __init__(self, url=DEFAULT_URL, model=DEFAULT_MODEL):
        self.url = url.rstrip("/") + "/chat/completions"
        self.model = model

    def __call__(
        self,
        messages,
        schema=None,
        max_tokens=4096,
        temperature=0.0,
        thinking=False,
        timeout=60.0,
    ):
        """Reply to `messages`; `schema` constrains the content to that JSON schema."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)},
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise LlmError(f"HTTP {error.code} from {self.url}: {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as error:
            raise LlmError(f"{self.url}: {error}") from None
        try:
            content = data["choices"][0]["message"].get("content") or ""
            return Reply(content, time.monotonic() - started)
        except (KeyError, IndexError, TypeError):
            raise LlmError(f"Unexpected reply from {self.url}: {data!r:.300}") from None

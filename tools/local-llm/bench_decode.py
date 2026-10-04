#!/usr/bin/env python3
"""OpenAI互換サーバーのツール呼び出しと生成速度を確認する。

使い方: OPENAI_API_KEY=<キー> python3 tools/local-llm/bench_decode.py http://10.99.0.1:8080/v1
ツール呼び出しを1回確認した後、思考なし・temperature 0.6・最大512トークンで、
コード2題・英語の文章1題・日本語の文章1題を各2回ストリーミングし、
最初のトークンから最後のトークンまでの生成速度（TTFTを除く）の中央値を出す。
"""

import json
import os
import statistics
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000/v1"
LABEL = sys.argv[2] if len(sys.argv) > 2 else BASE
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("OPENAI_API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]

PROMPTS = {
    "code": [
        "Write a Python implementation of an LRU cache class with get/put, type hints and docstrings, plus pytest tests.",
        "Write a TypeScript function that parses a CSV string with quoted fields and escaped quotes into an array of objects keyed by the header row. Include unit tests.",
    ],
    "english": [
        "Explain the difference between processes and threads to a junior developer, with practical advice on when to use each.",
    ],
    "japanese": [
        "日本の四季それぞれの特徴と、旅行するならおすすめの過ごし方を説明してください。",
    ],
}


def stream(model, prompt):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=HEADERS,
    )
    t0 = time.time()
    first = last = None
    chunks = 0
    usage_tokens = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage_tokens = d["usage"].get("completion_tokens")
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                ):
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
    n = usage_tokens or chunks
    return first - t0, n, n / (last - first) if last and last > first else 0.0


def check_tool_call(model):
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Open src/main.py and tell me what it does."}
        ],
        "tools": [tool],
        "max_tokens": 2048,
    }
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers=HEADERS,
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        calls = json.load(r)["choices"][0]["message"].get("tool_calls") or []
    names = [f'{c["function"]["name"]}({c["function"]["arguments"]})' for c in calls]
    print(f"[{LABEL}] tool call: {', '.join(names) if names else 'NONE'}")


def main():
    model = json.load(
        urllib.request.urlopen(
            urllib.request.Request(BASE + "/models", headers=HEADERS)
        )
    )["data"][0]["id"]
    print(f"[{LABEL}] model={model}")
    check_tool_call(model)
    stream(model, "Say hi.")  # warm-up
    for kind, prompts in PROMPTS.items():
        rates, ttfts = [], []
        for p in prompts:
            for _ in range(2):
                ttft, n, rate = stream(model, p)
                rates.append(rate)
                ttfts.append(ttft)
        print(
            f"[{LABEL}] {kind}: decode median {statistics.median(rates):.1f} tok/s "
            f"(min {min(rates):.1f}, max {max(rates):.1f}), TTFT median {statistics.median(ttfts):.2f}s"
        )


if __name__ == "__main__":
    main()

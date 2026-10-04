#!/usr/bin/env python3
"""OpenAI互換サーバーのツール呼び出しと生成速度を確認する。

使い方: python3 tools/local-llm/bench_decode.py [Base URL（既定 http://10.99.0.1:8080/v1）]
APIキーが必要なサーバーでは OPENAI_API_KEY に入れる。
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

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://10.99.0.1:8080/v1").rstrip("/")
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("OPENAI_API_KEY"):
    HEADERS["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
RUNS = 2

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

READ_FILE_TOOL = {
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


def request(path, body=None):
    """GET（bodyなし）またはJSONをPOSTし、レスポンスを返す。"""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers=HEADERS)
    return urllib.request.urlopen(req, timeout=600)


def stream(model, prompt):
    """1回ストリーミングし、(TTFT秒, 生成速度tok/s) を返す。"""
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
    t0 = time.perf_counter()
    first = last = None
    chunks = 0
    tokens = None
    with request("/chat/completions", body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens")
            for c in d.get("choices", []):
                delta = c.get("delta", {})
                if (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("reasoning")
                ):
                    last = time.perf_counter()
                    first = first or last
                    chunks += 1
    if first is None:
        raise RuntimeError(f"no output for: {prompt}")
    rate = (tokens or chunks) / (last - first) if last > first else 0.0
    return first - t0, rate


def tool_calls(model):
    """ファイルを読む依頼に対して返ったツール呼び出しを "名前(引数)" のリストで返す。"""
    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Open src/main.py and tell me what it does."}
        ],
        "tools": [READ_FILE_TOOL],
        "max_tokens": 2048,
    }
    with request("/chat/completions", body) as r:
        calls = json.load(r)["choices"][0]["message"].get("tool_calls") or []
    return [f"{c['function']['name']}({c['function']['arguments']})" for c in calls]


def main():
    with request("/models") as r:
        model = json.load(r)["data"][0]["id"]
    print(f"model: {model}")
    print(f"tool call: {', '.join(tool_calls(model)) or 'NONE'}")
    stream(model, "Say hi.")  # warm-up
    for kind, prompts in PROMPTS.items():
        ttfts, rates = zip(*(stream(model, p) for p in prompts for _ in range(RUNS)))
        print(
            f"{kind}: decode median {statistics.median(rates):.1f} tok/s "
            f"(min {min(rates):.1f}, max {max(rates):.1f}), "
            f"TTFT median {statistics.median(ttfts):.2f}s"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Talk to the audio models on a GPU host from a Mac, through a page in the browser."""

from __future__ import annotations

import argparse
import http.server
import json
from pathlib import Path
import queue
import signal
import sys
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent))

from session import (  # noqa: E402
    LANGUAGE_NAMES,
    SSH_PORT,
    TARGET_NAMES,
    Events,
    Options,
    Session,
)

HERE = Path(__file__).parent
PAGE = HERE / "index.html"
SPEECH_PAGE = HERE / "speech.html"
ASSETS = {
    "/page.css": (HERE / "page.css", "text/css; charset=utf-8"),
    "/page.js": (HERE / "page.js", "text/javascript; charset=utf-8"),
}
UI_PORT = 8766
SPEECH_UI_PORT = 8767  # so that both pages can run at once
HEARTBEAT_S = 15.0
MAX_BODY = 64 * 1024


def make_handler(session: Session, defaults: dict, port_host: str, page: Path):
    routes = {
        "/api/connect": lambda b: session.connect(
            b["target"], b["language"], b.get("user", ""), b.get("device", "")
        ),
        "/api/disconnect": lambda b: session.disconnect(),
        "/api/language": lambda b: session.set_language(b["language"]),
        "/api/hotwords": lambda b: session.set_hotwords(b["words"]),
        "/api/speak": lambda b: session.speak(b["text"], b.get("voice", "")),
        "/api/stop": lambda b: session.stop(),
    }

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _trusted(self):
            # Refuses pages from other sites that reach this port by name.
            return self.headers.get("Host") == port_host

        def _reply(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._reply(status, body, "application/json; charset=utf-8")

        def do_GET(self):
            if not self._trusted():
                self._json(403, {"error": "forbidden"})
            elif self.path == "/":
                self._reply(200, page.read_bytes(), "text/html; charset=utf-8")
            elif self.path in ASSETS:
                asset, content_type = ASSETS[self.path]
                self._reply(200, asset.read_bytes(), content_type)
            elif self.path == "/api/defaults":
                self._json(200, defaults)
            elif self.path == "/events":
                self._stream()
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(min(length, MAX_BODY))
            # A cross-site form cannot send this content type without a preflight.
            json_body = self.headers.get("Content-Type") == "application/json"
            route = routes.get(self.path)
            if not self._trusted() or not json_body:
                self._json(403, {"error": "forbidden"})
            elif route is None:
                self._json(404, {"error": "not found"})
            else:
                try:
                    route(json.loads(body or b"{}"))
                    self._json(200, {})
                except (KeyError, TypeError, ValueError) as exc:
                    self._json(400, {"error": str(exc)})

        def _stream(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            subscriber = session.events.subscribe()
            try:
                while True:
                    try:
                        event = subscriber.get(timeout=HEARTBEAT_S)
                        line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except queue.Empty:
                        line = ": heartbeat\n\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
            except OSError:
                pass  # the page went away
            finally:
                session.events.unsubscribe(subscriber)

    return Handler


def make_server(session, defaults, port, page=PAGE):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), None)
    server.daemon_threads = True
    host = f"127.0.0.1:{server.server_address[1]}"
    server.RequestHandlerClass = make_handler(session, defaults, host, page)
    return server, f"http://{host}/"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=sorted(TARGET_NAMES), default="dgx")
    parser.add_argument("--language", choices=tuple(LANGUAGE_NAMES), default="en")
    parser.add_argument("--user", default="", help="directory-access Unix username")
    parser.add_argument("--host", default="", help="SSH host or config alias")
    parser.add_argument("--port", type=int, default=SSH_PORT, help="0: the SSH config")
    parser.add_argument("--identity-file", default="")
    parser.add_argument("--output-device", default="", help="name or index")
    parser.add_argument("--capture-rate", type=int, default=Options.capture_rate)
    parser.add_argument(
        "--startup-timeout", type=float, default=Options.startup_timeout
    )
    parser.add_argument("--timeout", type=float, default=Options.timeout)
    parser.add_argument("--speech", action="store_true", help="the speech-only page")
    parser.add_argument("--ui-port", type=int, default=0, help="the page's port")
    parser.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser


STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def interrupt(signum, frame):
    # uv forwards the signal too: a second one must not cut the cleanup short.
    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, signal.SIG_IGN)
    raise KeyboardInterrupt


def main(argv=None):
    args = build_parser().parse_args(argv)
    # Closing the terminal must also release the remote GPU container.
    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, interrupt)
    session = Session(
        Options(
            host=args.host,
            port=args.port,
            identity_file=args.identity_file,
            output_device=args.output_device,
            capture_rate=args.capture_rate,
            startup_timeout=args.startup_timeout,
            timeout=args.timeout,
            listen=not args.speech,
        ),
        Events(),
    )
    defaults = {
        "target": args.target,
        "language": args.language,
        "user": args.user,
        "user_required": not args.host,
        "targets": TARGET_NAMES,
        "languages": LANGUAGE_NAMES,
    }
    page, port = (SPEECH_PAGE, SPEECH_UI_PORT) if args.speech else (PAGE, UI_PORT)
    server, url = make_server(session, defaults, args.ui_port or port, page)
    print(f"[audio] Open {url}  (Ctrl+C to quit)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("[audio] Closing", flush=True)
        session.close()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Web GUI for subgoal task runners: live progress, partial runs and recovery.

It also serves the robot's touch screen under /display/ (aha_tasks.display).
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

HTML = "text/html; charset=utf-8"
JS = "text/javascript; charset=utf-8"
CSS = "text/css; charset=utf-8"
# URL path -> (file under web/, content type). /display/ needs a screen.
WEB_FILES = {
    "/": ("index.html", HTML),
    "/app.js": ("app.js", JS),
    "/style.css": ("style.css", CSS),
    "/display/": ("display/index.html", HTML),
    "/display/display.js": ("display/display.js", JS),
    "/display/display.css": ("display/display.css", CSS),
}
DISPLAY = "/display"
IMAGE_PREFIX = "/display/api/image/"
COMMAND_FIELDS = {"command": str, "subgoals": list, "on_failure": str, "target": str}
HEARTBEAT = 15.0  # seconds between SSE keepalives


class StatusHub:
    """Latest runner status; SSE clients wait for its version to change."""

    def __init__(self, **payload):
        self._changed = threading.Condition()
        self.version = 0
        self.payload = payload

    def update(self, **changes):
        with self._changed:
            payload = dict(self.payload, **changes)
            if payload != self.payload:
                self.payload = payload
                self.version += 1
                self._changed.notify_all()

    def wait(self, version, timeout):
        with self._changed:
            self._changed.wait_for(lambda: self.version != version, timeout)
            return self.version, self.payload


def parse_command(body):
    data = json.loads(body or b"{}")
    if not isinstance(data, dict) or not isinstance(data.get("command"), str):
        raise ValueError("command is required")
    for key, kind in COMMAND_FIELDS.items():
        if key in data and not isinstance(data[key], kind):
            raise ValueError(f"{key} must be {kind.__name__}")
    if not all(isinstance(item, str) for item in data.get("subgoals", [])):
        raise ValueError("subgoals must be strings")
    return data


def parse_touch(body):
    """{"question": id, "choice": index} from the screen page."""
    data = json.loads(body or b"{}")
    if not isinstance(data, dict):
        raise ValueError("a JSON object is required")
    for key in ("question", "choice"):
        if type(data.get(key)) is not int:
            raise ValueError(f"{key} must be an integer")
    return data["question"], data["choice"]


def make_server(host, port, hub, send_command, web_dir, screen=None):
    """``send_command(dict) -> (accepted, message)`` forwards to the runner.

    With ``screen`` (aha_tasks.display.Screen) the robot's screen is at /display/.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                return self._json(200, hub.payload)
            if path == "/api/events":
                return self._events(hub)
            if path.startswith(DISPLAY) and screen is None:
                return self._json(404, {"error": "no screen"})
            if path == DISPLAY:
                self.send_response(301)
                self.send_header("Location", "display/")
                self.send_header("Content-Length", "0")
                return self.end_headers()
            if path == "/display/api/events":
                return self._events(screen.hub, "screen")
            if path.startswith(IMAGE_PREFIX):
                return self._image(path[len(IMAGE_PREFIX) :])
            if path not in WEB_FILES:
                return self._json(404, {"error": "not found"})
            name, content_type = WEB_FILES[path]
            self._send(200, (Path(web_dir) / name).read_bytes(), content_type)

        def do_POST(self):
            if self.path == "/api/command":
                return self._command()
            if self.path == "/display/api/touch" and screen is not None:
                return self._touch()
            self._json(404, {"error": "not found"})

        def _body(self):
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _command(self):
            try:
                command = parse_command(self._body())
            except ValueError as error:  # includes JSON decoding errors
                return self._json(400, {"accepted": False, "message": str(error)})
            accepted, message = send_command(command)
            self._json(200, {"accepted": accepted, "message": message})

        def _touch(self):
            # A JSON body needs a CORS preflight, so other sites cannot touch.
            content_type = self.headers.get("Content-Type") or ""
            if content_type.split(";")[0].strip() != "application/json":
                return self._json(
                    415, {"accepted": False, "message": "JSON body required"}
                )
            try:
                question, choice = parse_touch(self._body())
            except ValueError as error:
                return self._json(400, {"accepted": False, "message": str(error)})
            accepted, message = screen.touch(question, choice)
            self._json(200, {"accepted": accepted, "message": message})

        def _image(self, number):
            image = screen.image(int(number)) if number.isdecimal() else None
            if image is None:
                return self._json(404, {"error": "no such image"})
            mime, data = image
            self._send(200, data, mime)

        def _events(self, hub, key=None):
            """Stream the hub's payload, or only its `key` entry when that changes."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            version, sent, written = None, None, 0.0
            try:
                while True:
                    # Wake for the next keepalive even if other entries keep changing.
                    since = time.monotonic() - written
                    latest, payload = hub.wait(version, max(0.0, HEARTBEAT - since))
                    version = latest
                    data = payload if key is None else payload.get(key)
                    if written == 0.0 or data != sent:
                        sent = data
                        data = json.dumps(data, ensure_ascii=False)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    elif time.monotonic() - written >= HEARTBEAT:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        continue
                    self.wfile.flush()
                    written = time.monotonic()
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._send(code, body, "application/json; charset=utf-8")

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


class RunnerBridge:
    """ROS side: status subscription and the command service client."""

    def __init__(self, node, hub):
        from aha_msgs.srv import TaskCommand
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        from aha_tasks.runner import COMMAND_SERVICE, STATUS_TOPIC

        self.hub = hub
        self.request_type = TaskCommand.Request
        self.client = node.create_client(TaskCommand, COMMAND_SERVICE)
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        node.create_subscription(String, STATUS_TOPIC, self._on_status, qos)
        node.create_timer(0.5, self._check_online)

    def _on_status(self, message):
        self.hub.update(status=json.loads(message.data))

    def _check_online(self):
        self.hub.update(online=self.client.service_is_ready())

    def send(self, command, timeout=5.0):
        if not self.client.service_is_ready():
            return False, "Task runner is not available"
        future = self.client.call_async(
            self.request_type(
                command=command["command"],
                subgoals=command.get("subgoals", []),
                on_failure=command.get("on_failure", ""),
                target=command.get("target", ""),
            )
        )
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() > deadline:
                return False, "Task runner did not respond"
            time.sleep(0.01)
        response = future.result()
        return response.accepted, response.message


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("AHA_GUI_HOST", "127.0.0.1"),
        help="Bind address; default AHA_GUI_HOST or loopback",
    )
    parser.add_argument("--port", type=int, default=8765)
    args, ros_args = parser.parse_known_args(argv)

    from ament_index_python.packages import get_package_share_directory
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    from aha_tasks.display import Screen, ScreenBridge

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("task_monitor")
    hub = StatusHub(online=False, status=None)
    bridge = RunnerBridge(node, hub)
    screen = Screen(hub)  # one SSE stream carries the status and the screen
    display_node = Node("robot_display")
    screen_bridge = ScreenBridge(display_node, screen)
    web_dir = Path(get_package_share_directory("aha_tasks")) / "web"
    server = make_server(args.host, args.port, hub, bridge.send, web_dir, screen)
    executor = MultiThreadedExecutor()  # a question waits for a touch in a callback
    executor.add_node(node)
    executor.add_node(display_node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    node.get_logger().info(
        f"Task monitor on http://localhost:{args.port}/, "
        f"robot screen on http://localhost:{args.port}/display/"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        screen_bridge.close()  # an open question would keep executor.shutdown() waiting
        executor.shutdown()
        spinner.join(timeout=5.0)  # before tearing down the nodes
        display_node.destroy_node()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0

"""Web GUI for subgoal task runners: live progress, partial runs and recovery."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time

WEB_FILES = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}
COMMAND_FIELDS = {"command": str, "subgoals": list, "on_failure": str, "target": str}
HEARTBEAT = 15.0  # seconds between SSE keepalives


class StatusHub:
    """Latest runner status; SSE clients wait for its version to change."""

    def __init__(self):
        self._changed = threading.Condition()
        self.version = 0
        self.payload = {"online": False, "status": None}

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


def make_server(host, port, hub, send_command, web_dir):
    """``send_command(dict) -> (accepted, message)`` forwards to the runner."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/status":
                return self._json(200, hub.payload)
            if path == "/api/events":
                return self._events()
            name = "index.html" if path == "/" else path.lstrip("/")
            if name not in WEB_FILES:
                return self._json(404, {"error": "not found"})
            self._send(200, (Path(web_dir) / name).read_bytes(), WEB_FILES[name])

        def do_POST(self):
            if self.path != "/api/command":
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                command = parse_command(self.rfile.read(length))
            except ValueError as error:  # includes JSON decoding errors
                return self._json(400, {"accepted": False, "message": str(error)})
            accepted, message = send_command(command)
            self._json(200, {"accepted": accepted, "message": message})

        def _events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            version = None
            try:
                while True:
                    latest, payload = hub.wait(version, HEARTBEAT)
                    if latest == version:
                        self.wfile.write(b": keepalive\n\n")
                    else:
                        version = latest
                        data = json.dumps(payload, ensure_ascii=False)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
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
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("task_monitor")
    hub = StatusHub()
    bridge = RunnerBridge(node, hub)
    web_dir = Path(get_package_share_directory("aha_tasks")) / "web"
    server = make_server(args.host, args.port, hub, bridge.send, web_dir)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, daemon=True)
    spinner.start()
    node.get_logger().info(f"Task monitor on http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        executor.shutdown()
        spinner.join(timeout=5.0)  # before tearing down the node
        node.destroy_node()
        rclpy.try_shutdown()
    return 0

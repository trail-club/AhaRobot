"""Nonblocking observation/action IPC with an independently launched policy."""

import json
import os
import socket
import subprocess
import sys
import time

from .policy_protocol import decode_action, encode, observation_message
from .processes import policy_partition, stop_group


class PolicyProcess:
    def __init__(
        self,
        name,
        task_id,
        instruction,
        domain_id,
        directory,
        partition,
        *,
        startup_timeout=120,
        action_timeout=30,
    ):
        if domain_id == int(os.environ.get("ROS_DOMAIN_ID", "0")):
            raise ValueError("Policy and evaluator ROS domains must differ")
        self.partition = policy_partition(partition)
        self.action_timeout = action_timeout
        self.ready = False
        self.pending = None
        self.phase = None
        self.incoming = bytearray()
        self.outgoing = bytearray()
        self.log = (directory / "policy.log").open("w")
        self.connection, child = socket.socketpair()
        self.connection.setblocking(False)
        environment = dict(os.environ)
        environment.update(
            {
                "ROS_DOMAIN_ID": str(domain_id),
                "GZ_PARTITION": self.partition,
                "IGN_PARTITION": self.partition,
                "ROS_LOG_DIR": str(directory / "policy_ros_logs"),
                "ROS_HOME": str(directory / "policy_ros_home"),
                "GZ_SIM_LOG_PATH": str(directory / "policy_gz_logs"),
            }
        )
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "aha_sim_tasks.policy_worker",
                    "--fd",
                    str(child.fileno()),
                    "--policy",
                    name,
                ],
                pass_fds=(child.fileno(),),
                env=environment,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            self.connection.close()
            self.log.close()
            raise
        finally:
            child.close()
        self.request(
            {"op": "reset", "task_id": task_id, "instruction": instruction},
            "ready",
            startup_timeout,
        )

    def request(self, message, response, timeout):
        if self.pending is not None:
            raise RuntimeError("Policy already has a pending request")
        self.outgoing.extend(encode(message))
        self.pending = response
        self.deadline = time.monotonic() + timeout

    def act(self, observation):
        if not self.ready:
            raise RuntimeError("Policy has not completed initialization")
        self.request(observation_message(observation), "action", self.action_timeout)

    def poll(self):
        if self.pending is None:
            if self.process.poll() is not None:
                raise RuntimeError("Policy worker exited; see policy.log")
            return None
        if time.monotonic() >= self.deadline:
            raise TimeoutError(
                f"Policy {self.pending} response timed out; see policy.log"
            )
        try:
            if self.outgoing:
                sent = self.connection.send(self.outgoing)
                del self.outgoing[:sent]
        except BlockingIOError:
            pass
        except OSError as error:
            raise RuntimeError("Policy worker disconnected; see policy.log") from error
        # Read only bounded chunks per iteration so ROS and scoring keep running.
        for _ in range(4):
            try:
                chunk = self.connection.recv(65536)
            except BlockingIOError:
                break
            if not chunk:
                if b"\n" not in self.incoming:
                    raise RuntimeError("Policy worker exited; see policy.log")
                break
            self.incoming.extend(chunk)
            if b"\n" in self.incoming:
                break
        if b"\n" in self.incoming:
            line, _, remainder = self.incoming.partition(b"\n")
            self.incoming = bytearray(remainder)
            response = json.loads(line)
            if response["op"] == "error":
                raise RuntimeError(
                    f"Policy worker: {response['reason']}; see policy.log"
                )
            if response["op"] != self.pending:
                raise RuntimeError(f"Unexpected policy response: {response['op']}")
            self.pending = None
            if response["op"] == "ready":
                self.ready = True
                return None
            action = decode_action(response["action"])
            self.phase = response.get("phase")
            return action
        return None

    def close(self):
        self.connection.close()
        try:
            # An idle worker exits on IPC EOF without interrupting user code.
            try:
                self.process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            stop_group(self.process, self.partition, timeout=1)
        finally:
            self.log.close()

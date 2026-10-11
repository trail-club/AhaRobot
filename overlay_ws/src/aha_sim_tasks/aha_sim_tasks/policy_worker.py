"""Execute team-owned policies without evaluator memory or simulation transports."""

import argparse
import json
import socket
import traceback

from .policies import load_policy
from .policy_protocol import action_message, decode_observation, encode


def serve(connection, name):
    # Import and factory execution happen here, never in the evaluator.
    with connection, connection.makefile("rb") as reader:
        try:
            policy = load_policy(name)
            for line in reader:
                request = json.loads(line)
                if request["op"] == "reset":
                    policy.reset(request["task_id"], request["instruction"])
                    response = {"op": "ready"}
                elif request["op"] == "act":
                    action = policy.act(decode_observation(request["observation"]))
                    phase = getattr(policy, "phase", None)
                    response = {
                        "op": "action",
                        "action": action_message(action),
                        "phase": phase if isinstance(phase, str) else None,
                    }
                else:
                    raise ValueError(f"Unknown policy request: {request['op']}")
                connection.sendall(encode(response))
        except Exception as error:
            traceback.print_exc()
            try:
                connection.sendall(
                    encode(
                        {"op": "error", "reason": f"{type(error).__name__}: {error}"}
                    )
                )
            except OSError:
                pass
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    try:
        return serve(socket.socket(fileno=args.fd), args.policy)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

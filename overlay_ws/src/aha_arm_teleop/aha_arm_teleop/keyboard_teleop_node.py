"""両腕のキーボード教示ノード。

`ros2 launch` はキーを渡さない。対話シェルで `ros2 run aha_arm_teleop keyboard_teleop`
する。手順は docs/arm-keyboard-teleop.md。
"""

from __future__ import annotations

import sys
import time

import rclpy
import rclpy.qos
import sensor_msgs.msg
import std_msgs.msg
from astra_controller_interfaces.msg import JointCommand

from .logic import HELP, Teleop


class KeyboardTeleopNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("keyboard_arm_teleop")
        self.declare_parameter("step", 8)
        self.declare_parameter("rate", 250)
        self.declare_parameter("lead", 80)
        self.declare_parameter("stall_time", 0.5)
        self.declare_parameter("max_jump", 250)
        self.declare_parameter("sides", ["left", "right"])
        self.declare_parameter("joint_states_topic", "/joint_states")

        sides = tuple(self.get_parameter("sides").value)
        unknown = [s for s in sides if s not in ("left", "right")]
        if unknown:
            raise ValueError(f"sides は left / right だけ: {unknown}")
        self.teleop = Teleop(
            step=int(self.get_parameter("step").value),
            rate=int(self.get_parameter("rate").value),
            lead=int(self.get_parameter("lead").value),
            stall_time=float(self.get_parameter("stall_time").value),
            max_jump=int(self.get_parameter("max_jump").value),
            sides=sides,
        )
        qos = rclpy.qos.qos_profile_sensor_data
        self._arm_pub = {}
        self._grip_pub = {}
        self._torque_pub = {}
        for side in sides:
            self._arm_pub[side] = self.create_publisher(
                JointCommand, f"/{side}/arm/joint_command", qos
            )
            self._grip_pub[side] = self.create_publisher(
                JointCommand, f"/{side}/arm/gripper_joint_command", qos
            )
            self._torque_pub[side] = self.create_publisher(
                std_msgs.msg.UInt8, f"/{side}/arm/torque_enable", 10
            )
        self.create_subscription(
            sensor_msgs.msg.JointState,
            self.get_parameter("joint_states_topic").value,
            self._on_joint_states,
            10,
        )
        self._last_draw = 0.0

    def _on_joint_states(self, msg: sensor_msgs.msg.JointState) -> None:
        for name, position in zip(msg.name, msg.position):
            self.teleop.update_feedback(name, position)

    def publish(self, outputs) -> None:
        for command in outputs.commands:
            msg = JointCommand()
            msg.name = list(command.names)
            msg.position_cmd = list(command.positions)
            if command.topic.endswith("gripper_joint_command"):
                side = "left" if "/left/" in command.topic else "right"
                self._grip_pub[side].publish(msg)
            else:
                side = "left" if "/left/" in command.topic else "right"
                self._arm_pub[side].publish(msg)
        for torque in outputs.torque:
            side = "left" if "/left/" in torque.topic else "right"
            if side not in self._torque_pub:
                continue
            self._torque_pub[side].publish(std_msgs.msg.UInt8(data=int(torque.value)))

    def draw(self, status: str, now: float) -> None:
        if now - self._last_draw < 0.1:
            return
        self._last_draw = now
        sys.stdout.write(f"\r\033[K{status}")
        sys.stdout.flush()

    def release(self) -> None:
        """終了時に両腕のトルクを切る。プロセスが死んでもサーボが押し続けないように。"""
        for side, pub in self._torque_pub.items():
            pub.publish(std_msgs.msg.UInt8(data=0))
            self.get_logger().info(f"{side} torque_enable 0")


def main(args=None) -> None:
    try:
        import select
        import termios
        import tty
    except ImportError:
        sys.exit("このノードは Linux の対話端末で動かす。Windows のコンソールではキーを取れない。")

    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    if not sys.stdin.isatty():
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(
            "キーボードがこのプロセスに繋がっていない。\n"
            "ros2 launch ではなく、対話シェルで ros2 run aha_arm_teleop keyboard_teleop する。"
        )

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    print(HELP)
    print("joint_states が来るまで方向キーは無視する。")
    last = time.time()
    try:
        tty.setcbreak(fd)
        while rclpy.ok() and not node.teleop.quit:
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.time()
            dt, last = now - last, now
            keys = _drain_keys(select)
            outputs = node.teleop.tick(keys, now, dt)
            node.publish(outputs)
            if outputs.banner:
                sys.stdout.write("\n" + outputs.banner + "\n")
                sys.stdout.flush()
            node.draw(outputs.status, now)
            time.sleep(0.005)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        node.release()
        # トルク OFF が送られる前にプロセスが終わらないように、1 回だけ回す。
        rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        rclpy.shutdown()
        print("\n両腕トルクOFF。終了しました。")


def _drain_keys(select_mod) -> list[str]:
    keys = []
    while select_mod.select([sys.stdin], [], [], 0)[0]:
        char = sys.stdin.read(1)
        if not char:
            break
        keys.append(char)
        if len(keys) > 200:
            break
    return keys


if __name__ == "__main__":
    main()

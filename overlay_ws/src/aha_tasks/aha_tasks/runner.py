"""Simulation task entry point and RViz location labels."""

import argparse
import math
from pathlib import Path

from aha_tasks.flow import execute_flow, load_flow


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return number


def publish_labels(node, flow):
    from rclpy.qos import DurabilityPolicy, QoSProfile
    from visualization_msgs.msg import Marker, MarkerArray

    publisher = node.create_publisher(
        MarkerArray,
        "/aha/tasks/locations",
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    markers = MarkerArray()
    for i, (name, (x, y, yaw)) in enumerate(flow["locations"].items()):
        for text in (False, True):
            marker = Marker()
            marker.header.frame_id = flow["frame_id"]
            marker.ns = "labels" if text else "poses"
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING if text else Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x, marker.pose.position.y = float(x), float(y)
            marker.pose.position.z = 0.6 if text else 0.1
            marker.pose.orientation.z = math.sin(yaw / 2)
            marker.pose.orientation.w = math.cos(yaw / 2)
            marker.scale.x, marker.scale.y, marker.scale.z = (
                (0.0, 0.0, 0.22) if text else (0.45, 0.08, 0.08)
            )
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (
                0.2,
                1.0,
                0.3,
                1.0,
            )
            marker.text = name if text else ""
            markers.markers.append(marker)
    publisher.publish(markers)
    return publisher


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the flow without ROS or motion"
    )
    parser.add_argument("--startup-timeout", type=positive_seconds, default=300.0)
    parser.add_argument("--goal-timeout", type=positive_seconds, default=600.0)
    args, ros_args = parser.parse_known_args(argv)
    flow = load_flow(args.config)
    if args.dry_run:
        print("DRY RUN: no navigation goals will be sent")
        for step in flow["steps"]:
            if "navigate" in step:
                label = step["navigate"]
                print(f"{step['id']}: navigate {label} {flow['locations'][label]}")
            else:
                print(f"{step['id']}: SKIPPED {step['skip']}")
        return 0

    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from aha_tasks.navigation import Navigation

    # Keep ROS alive during KeyboardInterrupt so the active goal can be cancelled.
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = Node("human_interaction")
    navigation = Navigation(node, args.startup_timeout, args.goal_timeout)
    code = 1
    try:
        if not node.get_parameter("use_sim_time").value:
            raise ValueError("This runner requires use_sim_time:=true")
        publish_labels(node, flow)
        node.get_logger().info("Waiting for active Nav2")
        navigation.ready()
        code = (
            0 if execute_flow(flow, navigation.navigate, node.get_logger().info) else 1
        )
    except KeyboardInterrupt:
        code = 130
    except Exception as error:
        node.get_logger().error(str(error))
    finally:
        try:
            navigation.cancel()
        except Exception as error:
            node.get_logger().error(
                f"Could not confirm navigation cancellation: {error}"
            )
            code = 1
        node.destroy_node()
        rclpy.try_shutdown()
    return code

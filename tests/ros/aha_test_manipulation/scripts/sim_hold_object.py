#!/usr/bin/env python3
"""Simulation only: weld a new box / cylinder into a gripper (DetachableJoint).

  ros2 run aha_test_manipulation sim_hold_object.py --size 0.06 0.06 0.12
  ros2 run aha_test_manipulation sim_hold_object.py --arm left --shape cylinder \
      --size 0.065 0.065 0.12 --grasp top
  ros2 run aha_test_manipulation sim_hold_object.py --release   # detach and delete

Detaches and deletes the previous object, moves the arm to --state (default
carry), opens the gripper, spawns aha_held_object_<arm> at the grasp pose of
aha_test_manipulation.place_geometry, attaches it and closes the fingers
(aha_test_manipulation.sim_held_object). Needs the sim with use_manip:=true and
GZ_PARTITION of that sim.
"""

import argparse
import os
import sys

import rclpy
from rclpy.parameter import Parameter

from aha_test_manipulation import place_geometry as geo
from aha_test_manipulation.sim_held_object import GzSim, SimHolder, model_name, tilt_deg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--arm", choices=["right", "left"], default="right")
    parser.add_argument("--shape", choices=["box", "cylinder"], default="box")
    parser.add_argument("--size", type=float, nargs=3, default=[0.06, 0.06, 0.12])
    parser.add_argument("--grasp", choices=list(geo.GRASPS), default=geo.SIDE)
    parser.add_argument(
        "--state", default="carry", help="named arm state; empty = stay"
    )
    parser.add_argument("--tilt", type=float, default=geo.DEFAULT_TOP_TILT)
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument(
        "--world", default=None, help="Gazebo world name (default: detected)"
    )
    parser.add_argument(
        "--robot", default="aha_robot", help="Gazebo model name of the robot"
    )
    parser.add_argument(
        "--release", action="store_true", help="detach and delete the object"
    )
    args = parser.parse_args(rclpy.utilities.remove_ros_args(sys.argv)[1:])

    rclpy.init()
    node = rclpy.create_node(
        "sim_hold_object", parameter_overrides=[Parameter("use_sim_time", value=True)]
    )
    try:
        gz = GzSim(args.world)
        if args.release:
            gz.release(args.arm)
            print(f"released {model_name(args.arm)}")
            return 0
        holder = SimHolder(node, gz, args.robot)
        T = holder.hold(
            args.arm,
            args.shape,
            args.size,
            args.grasp,
            args.state or None,
            args.tilt,
            args.mass,
            args.mu,
        )
        x, y, z = T[:3, 3]
        print(
            f"{model_name(args.arm)} attached at world ({x:.3f}, {y:.3f}, {z:.3f}), "
            f"tilt {tilt_deg(T):.1f} deg"
        )
        return 0
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # The gz-transport Python binding sometimes crashes during interpreter teardown.
    os._exit(code)

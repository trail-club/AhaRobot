"""Launch the evaluation scene and robot, without a policy or scorer."""

import os

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    EnvironmentVariable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="true"),
            SetEnvironmentVariable(
                "GZ_SIM_SYSTEM_PLUGIN_PATH",
                [
                    os.path.join(get_package_prefix("aha_sim_tasks"), "lib"),
                    os.pathsep,
                    EnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", default_value=""),
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("aha_bringup"), "launch", "sim.launch.py"]
                    )
                ),
                launch_arguments={
                    "world_path": PathJoinSubstitution(
                        [FindPackageShare("aha_sim_tasks"), "worlds", "task_tables.sdf"]
                    ),
                    "headless": LaunchConfiguration("headless"),
                    "apple_pick_mode": "false",
                    "bridge_clock": "false",
                    "activate_controllers_as_group": "true",
                }.items(),
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                arguments=[
                    "/evaluation/poses@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
                    "/evaluation/grasped@std_msgs/msg/Bool[gz.msgs.Boolean",
                    "/evaluation/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
                ],
                remappings=[("/evaluation/clock", "/clock")],
                output="screen",
            ),
        ]
    )

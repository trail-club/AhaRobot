"""RViz manipulation view: MotionPlanning, cameras, control panel, goal error.

Included by manip.launch.py rviz:=true (sim.launch.py use_manip:=true
manip_rviz:=true). Runnable on its own against a simulation that already runs
move_group, so the view can be restarted without restarting Gazebo:

  ros2 launch aha_manipulation manip_view.launch.py

Nodes:
  rviz2                  rviz/manip.rviz; the MotionPlanning display reads
                         robot_description and robot_description_semantic
                         from their topics and needs only the kinematics here.
                         Joint states: /aha/manipulation/joint_states, the
                         same input as move_group (manip.launch.py)
  sim_control_panel.py   aha_perception's panel with arm_controls:=true
                         (base, head, right gripper, right lift); wall clock
  eef_goal_monitor.py    goal of the last plan vs. current link_ree;
                         wall clock (latest TF)
  camera_bridge          camera_bridge:=true only; aha_perception's head and
                         wrist camera bridge (use_perception:=true already
                         starts it)
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("aha_manipulation")
    perception = FindPackageShare("aha_perception")
    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)
    kinematics = yaml.safe_load(
        (
            Path(get_package_share_directory("aha_manipulation"))
            / "config/kinematics.yaml"
        ).read_text()
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution([share, "rviz", "manip.rviz"]),
            ),
            DeclareLaunchArgument(
                "control_panel",
                default_value="true",
                choices=["true", "false"],
                description="Start the control panel (base, head, gripper, lift)",
            ),
            DeclareLaunchArgument(
                "camera_bridge",
                default_value="false",
                choices=["true", "false"],
                description="Bridge the Gazebo cameras (off when aha_perception runs)",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", LaunchConfiguration("rviz_config")],
                parameters=[
                    {
                        "robot_description_kinematics": kinematics,
                        "use_sim_time": use_sim_time,
                    }
                ],
                remappings=[("joint_states", "/aha/manipulation/joint_states")],
                output="screen",
            ),
            Node(
                package="aha_perception",
                executable="sim_control_panel.py",
                parameters=[
                    PathJoinSubstitution([perception, "config", "head.yaml"]),
                    {"arm_controls": True},
                ],
                output="screen",
                condition=IfCondition(LaunchConfiguration("control_panel")),
            ),
            Node(
                package="aha_manipulation",
                executable="eef_goal_monitor.py",
                output="screen",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [perception, "launch", "camera_bridge.launch.py"]
                    )
                ),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time")
                }.items(),
                condition=IfCondition(LaunchConfiguration("camera_bridge")),
            ),
        ]
    )

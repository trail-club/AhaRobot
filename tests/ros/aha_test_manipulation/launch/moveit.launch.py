"""MoveIt 2 move_group (and optional RViz) for AhaRobot.

Expects robot_state_publisher, /joint_states and the arm / lift controllers
from aha_bringup/sim.launch.py. robot_description is generated from the
xacro and args in .setup_assistant (aha_test_description's test bed xacro,
sim:=true), the same as the simulation's robot_state_publisher.

Arguments:
  use_sim_time (true)   use /clock
  rviz         (true)   RViz with the MotionPlanning display
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def moveit_config():
    return (
        MoveItConfigsBuilder("aha_robot", package_name="aha_test_manipulation")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl", "pilz_industrial_motion_planner"])
        .planning_scene_monitor(publish_robot_description_semantic=True)
        .to_moveit_configs()
    )


def generate_launch_description():
    config = moveit_config()
    use_sim_time = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    rviz_config = str(
        Path(get_package_share_directory("aha_test_manipulation"))
        / "rviz"
        / "manip.rviz"
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        parameters=[config.to_dict(), use_sim_time],
        output="screen",
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="moveit_rviz",
        arguments=["-d", rviz_config],
        parameters=[
            config.robot_description,
            config.robot_description_semantic,
            config.robot_description_kinematics,
            config.planning_pipelines,
            config.joint_limits,
            use_sim_time,
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
        output="log",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz with the MotionPlanning display",
            ),
            move_group,
            rviz,
        ]
    )

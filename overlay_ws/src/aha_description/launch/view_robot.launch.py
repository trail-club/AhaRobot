"""Launch: robot_state_publisher + joint_state_publisher_gui + RViz.

For checking the URDF/xacro visually without Gazebo.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    xacro_path = PathJoinSubstitution([
        FindPackageShare("aha_description"), "urdf", "aha_robot.urdf.xacro",
    ])

    return LaunchDescription([
        DeclareLaunchArgument("sim", default_value="false"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": ParameterValue(
                    Command(["xacro ", xacro_path, " sim:=", LaunchConfiguration("sim")]),
                    value_type=str,
                ),
            }],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
        ),
    ])

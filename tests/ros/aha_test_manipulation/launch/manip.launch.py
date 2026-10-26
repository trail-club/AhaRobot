"""Test bed manipulation launch, next to aha_bringup/sim.launch.py started with
robot_xacro:=<aha_test_description's aha_robot_test.urdf.xacro> (tests/README.md).

Starts:
  - moveit.launch.py: move_group, and RViz unless rviz:=false
  - place_server:=true (default)     place_server.py (/aha/manipulation/place)
  - sim_held_object:=true (default)  ros_gz_bridge for the held-object topics of
    the URDF's DetachableJoint plugins (/aha/sim/held_object/<arm>/{attach,
    detach,state}); place_server.py releases through .../detach
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    IfElseSubstitution,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

HELD_OBJECT_TOPIC = "/aha/sim/held_object/{arm}"


def held_object_bridge_arguments():
    arguments = []
    for arm in ("right", "left"):
        base = HELD_OBJECT_TOPIC.format(arm=arm)
        arguments += [
            f"{base}/attach@std_msgs/msg/Empty]gz.msgs.Empty",
            f"{base}/detach@std_msgs/msg/Empty]gz.msgs.Empty",
            f"{base}/state@std_msgs/msg/String[gz.msgs.StringMsg",
        ]
    return arguments


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    sim_held_object = LaunchConfiguration("sim_held_object")
    moveit_launch = PathJoinSubstitution(
        [FindPackageShare("aha_test_manipulation"), "launch", "moveit.launch.py"]
    )

    held_object_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="held_object_bridge",
        arguments=held_object_bridge_arguments(),
        condition=IfCondition(sim_held_object),
        output="screen",
    )

    place_server = Node(
        package="aha_test_manipulation",
        executable="place_server.py",
        parameters=[
            {
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                "release_topic": IfElseSubstitution(
                    sim_held_object, HELD_OBJECT_TOPIC + "/detach", ""
                ),
            }
        ],
        condition=IfCondition(LaunchConfiguration("place_server")),
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz with the MotionPlanning display",
            ),
            DeclareLaunchArgument(
                "place_server",
                default_value="true",
                description="Start the PlaceObject action server",
            ),
            DeclareLaunchArgument(
                "sim_held_object",
                default_value="true",
                description="Bridge the Gazebo held-object topics and release through them",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(moveit_launch),
                launch_arguments={
                    "use_sim_time": use_sim_time,
                    "rviz": LaunchConfiguration("rviz"),
                }.items(),
            ),
            held_object_bridge,
            place_server,
        ]
    )

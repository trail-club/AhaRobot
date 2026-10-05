"""Real head (and camera) test: move the real head from the simulation UI,
without Gazebo.

The head servos are driven by aha_servo/servo_trajectory_bridge.py, which
serves the same /head_controller/joint_trajectory topic and
follow_joint_trajectory action as the simulation's JointTrajectoryController,
so the control panel, head_look_at.py and teleop_head.py work as in the
simulation.

  ros2 launch aha_perception real_head_camera.launch.py
  ros2 launch aha_perception real_head_camera.launch.py camera:=true
  ros2 launch aha_perception real_head_camera.launch.py port:=/dev/ttyUSB1 baud:=921600 rviz:=false

Starts:
  - robot_state_publisher (xacro sim:=false, wall clock)
  - joint_state_publisher: /joint_states with every movable joint; the head
    comes from /head_controller/joint_states, all other joints read 0
  - aha_servo servo_trajectory_bridge.py with config (default
    aha_servo/config/head.yaml; port/baud override it when given)
  - rviz:=true (default)  sim_view.launch.py with use_sim_time:=false,
                          fixed_frame:=base_link (RViz, control panel,
                          look-at; the base pad does nothing)
  - camera:=true          rosbridge_server on 127.0.0.1:9090 and
                          pointcloud.launch.py. The D435i stream
                          (stream_realsense.py through rosbridge) is started
                          separately, see aha_perception/README.md.

Do not run next to sim.launch.py on the same ROS_DOMAIN_ID: both serve
/head_controller and /joint_states.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# controller_name in aha_servo/config/head.yaml
HEAD_STATES = "/head_controller/joint_states"


def head_bridge(context):
    overrides = {}
    port = LaunchConfiguration("port").perform(context)
    baud = LaunchConfiguration("baud").perform(context)
    if port:
        overrides["port"] = port
    if baud:
        overrides["baud"] = int(baud)
    return [
        Node(
            package="aha_servo",
            executable="servo_trajectory_bridge.py",
            name="aha_head_servo_bridge",
            parameters=[LaunchConfiguration("config").perform(context), overrides],
            output="screen",
        )
    ]


def generate_launch_description():
    xacro_path = PathJoinSubstitution(
        [
            FindPackageShare("aha_description"),
            "urdf",
            "aha_robot.urdf.xacro",
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(
            Command(["xacro ", xacro_path, " sim:=false"]),
            value_type=str,
        ),
    }

    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[robot_description, {"use_sim_time": False}],
        output="screen",
    )

    jsp = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"source_list": [HEAD_STATES], "rate": 30}],
        output="screen",
    )

    sim_view = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aha_perception"),
                    "launch",
                    "sim_view.launch.py",
                ]
            )
        ),
        launch_arguments={"use_sim_time": "false", "fixed_frame": "base_link"}.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # Host network: bind to loopback so the bridge is not exposed on the LAN.
    rosbridge = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("rosbridge_server"),
                    "launch",
                    "rosbridge_websocket_launch.xml",
                ]
            )
        ),
        launch_arguments={"address": "127.0.0.1", "port": "9090"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )
    pointcloud = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("aha_perception"),
                    "launch",
                    "pointcloud.launch.py",
                ]
            )
        ),
        launch_arguments={"use_sim_time": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("aha_servo"),
                        "config",
                        "head.yaml",
                    ]
                ),
                description="Head servo bus config (aha_servo/joint_map.py format)",
            ),
            DeclareLaunchArgument(
                "port",
                default_value="",
                description="Serial port (empty: from config)",
            ),
            DeclareLaunchArgument(
                "baud",
                default_value="",
                description="Baud rate (empty: from config; 921600 for tools/firmware/right_arm/bridge)",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Start RViz, the control panel and the head look-at node",
            ),
            DeclareLaunchArgument(
                "camera",
                default_value="false",
                description="Start rosbridge (127.0.0.1:9090) and the point cloud node "
                "for the real D435i stream",
            ),
            rsp,
            jsp,
            OpaqueFunction(function=head_bridge),
            sim_view,
            rosbridge,
            pointcloud,
        ]
    )

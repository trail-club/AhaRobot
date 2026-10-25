"""Real robot on its frame (aha_description/frame.py): arms, head, head camera
and plane detection, without Gazebo.

  ros2 launch aha_bringup real.launch.py
  ros2 launch aha_bringup real.launch.py arms:=false head:=manual
  ros2 launch aha_bringup real.launch.py right_device:=/dev/ttyUSB2 left_device:=/dev/ttyUSB0

Starts:
  - robot_state_publisher with aha_robot.urdf.xacro (sim:=false) rewritten by
    aha_description.frame for frame_config; the lifts and wheels are fixed
  - static TF odom -> base_footprint (identity: the frame does not move), so
    plane detection tracks surfaces in odom as on the mobile robot
  - arms:=true   aha_bringup arms.launch.py (left/right arm_node on
                 left_device / right_device; record passed through)
  - head:=servo  aha_servo servo_trajectory_bridge.py on head_port
                 head:=manual  aha_perception manual_head.py (no servos)
                 head:=none    nothing; the head joints need another source
                 The head state goes to /joint_states next to the arms'.
  - camera:=true aha_perception realsense.launch.py and pointcloud.launch.py
  - planes:=true aha_perception plane_detection.launch.py (furniture_config)
  - rviz:=true   aha_perception sim_view.launch.py (fixed frame
                 base_footprint, control panel and head look-at)

Every joint state is published by the arms and the head on /joint_states;
there is no joint_state_publisher filling in zeros. Do not run next to
sim.launch.py on the same ROS_DOMAIN_ID.
"""

from pathlib import Path

import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aha_description.frame import frame_robot, load_frame

HEAD_STATES = "/head_controller/joint_states"  # aha_servo head.yaml controller_name


def _share(package: str) -> Path:
    return Path(get_package_share_directory(package))


def _launch(package: str, name: str):
    return PythonLaunchDescriptionSource(str(_share(package) / "launch" / name))


def robot_description(frame_config: str) -> str:
    """URDF of the robot on the frame described by ``frame_config``."""
    urdf = xacro.process_file(
        str(_share("aha_description") / "urdf" / "aha_robot.urdf.xacro"),
        mappings={"sim": "false"},
    ).toxml()
    return frame_robot(urdf, load_frame(frame_config))


def _robot(context):
    description = robot_description(LaunchConfiguration("frame_config").perform(context))
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": description, "use_sim_time": False}],
            output="screen",
        )
    ]


def _head(context):
    head = LaunchConfiguration("head").perform(context)
    remap = [(HEAD_STATES, "/joint_states")]
    if head == "servo":
        return [
            Node(
                package="aha_servo",
                executable="servo_trajectory_bridge.py",
                name="aha_head_servo_bridge",
                parameters=[
                    str(_share("aha_servo") / "config" / "head.yaml"),
                    {"port": LaunchConfiguration("head_port").perform(context)},
                ],
                remappings=remap,
                output="screen",
            )
        ]
    if head == "manual":
        return [
            Node(
                package="aha_perception",
                executable="manual_head.py",
                parameters=[str(_share("aha_perception") / "config" / "head.yaml")],
                remappings=remap,
                output="screen",
            )
        ]
    return []


def generate_launch_description():
    odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_tf",
        arguments=["--frame-id", "odom", "--child-frame-id", "base_footprint"],
        output="screen",
    )
    arms = IncludeLaunchDescription(
        _launch("aha_bringup", "arms.launch.py"),
        launch_arguments={
            "left_device": LaunchConfiguration("left_device"),
            "right_device": LaunchConfiguration("right_device"),
            "use_left": LaunchConfiguration("use_left"),
            "use_right": LaunchConfiguration("use_right"),
            "record": LaunchConfiguration("record"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("arms")),
    )
    camera = IncludeLaunchDescription(
        _launch("aha_perception", "realsense.launch.py"),
        condition=IfCondition(LaunchConfiguration("camera")),
    )
    pointcloud = IncludeLaunchDescription(
        _launch("aha_perception", "pointcloud.launch.py"),
        launch_arguments={"use_sim_time": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )
    planes = IncludeLaunchDescription(
        _launch("aha_perception", "plane_detection.launch.py"),
        launch_arguments={
            "furniture_config": LaunchConfiguration("furniture_config"),
        }.items(),
        condition=IfCondition(LaunchConfiguration("planes")),
    )
    rviz = IncludeLaunchDescription(
        _launch("aha_perception", "sim_view.launch.py"),
        launch_arguments={
            "use_sim_time": "false",
            "fixed_frame": "base_footprint",
        }.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "frame_config",
                default_value=str(
                    _share("aha_description") / "config" / "frame_table_202610.yaml"
                ),
                description="Frame model (aha_description/frame.py format)",
            ),
            DeclareLaunchArgument("arms", default_value="true"),
            DeclareLaunchArgument("use_left", default_value="true"),
            DeclareLaunchArgument("use_right", default_value="true"),
            DeclareLaunchArgument("left_device", default_value="/dev/ttyUSB2"),
            DeclareLaunchArgument("right_device", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument(
                "record",
                default_value="false",
                description="rosbag of every topic (arms.launch.py)",
            ),
            DeclareLaunchArgument(
                "head",
                default_value="servo",
                choices=["servo", "manual", "none"],
                description="servo: head servos on head_port; manual: commanded "
                "angles as the state (manual_head.py); none: no head node",
            ),
            DeclareLaunchArgument("head_port", default_value="/dev/ttyUSB1"),
            DeclareLaunchArgument("camera", default_value="true"),
            DeclareLaunchArgument("planes", default_value="true"),
            DeclareLaunchArgument(
                "furniture_config",
                default_value="",
                description="Furniture YAML for plane detection (empty: none)",
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            OpaqueFunction(function=_robot),
            odom,
            arms,
            OpaqueFunction(function=_head),
            camera,
            pointcloud,
            planes,
            rviz,
        ]
    )

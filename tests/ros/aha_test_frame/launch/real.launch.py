"""Real robot on its frame (aha_test_frame/frame.py): arms, head, head camera
and plane detection, without Gazebo.

  ros2 launch aha_test_frame real.launch.py
  ros2 launch aha_test_frame real.launch.py arms:=false head:=manual
  ros2 launch aha_test_frame real.launch.py use_left:=false right_device:=/dev/ttyUSB1

Starts:
  - robot_state_publisher with aha_robot.urdf.xacro (sim:=false) rewritten by
    aha_test_frame.frame for frame_config; the lifts and wheels are fixed
  - static TF odom -> base_footprint (identity: the frame does not move), so
    plane detection tracks surfaces in odom as on the mobile robot
  - joint_state_publisher: /joint_states from /head_controller/joint_states
    and the arms' /<side>/arm/joint_states; joints without a source read 0
  - arms:=true   aha_bringup arms.launch.py (left/right arm_node on
                 left_device / right_device; record passed through) and,
                 with trajectory_bridge, aha_manipulation
                 arm_trajectory_bridge.py per arm run by python3 (the
                 script is installed without the executable bit, which a
                 --symlink-install build keeps, so `ros2 run` cannot start it)
  - head:=servo  aha_servo servo_trajectory_bridge.py on head_port
                 head:=manual  aha_perception manual_head.py (no servos)
                 head:=imu     aha_perception imu_head.py: no servos, tilt from
                               the D435i IMU, pan from the commanded angle
                 head:=none    nothing (the head reads 0)
  - camera:=true aha_perception realsense.launch.py (IMU only for head:=imu)
                 and pointcloud.launch.py
  - planes:=true aha_perception plane_detection.launch.py (furniture_config)
  - rviz:=true   aha_perception sim_view.launch.py (fixed frame
                 base_footprint, control panel and head look-at)

Do not run next to sim.launch.py on the same ROS_DOMAIN_ID.
"""

from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    GroupAction,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from aha_test_frame.frame import frame_description

HEAD_STATES = "/head_controller/joint_states"  # aha_servo head.yaml controller_name
ARM_STATES = ["/left/arm/joint_states", "/right/arm/joint_states"]  # arms.launch.py


def _share(package: str) -> Path:
    return Path(get_package_share_directory(package))


def _launch(package: str, name: str):
    return PythonLaunchDescriptionSource(str(_share(package) / "launch" / name))


def _robot(context):
    description = frame_description(
        LaunchConfiguration("frame_config").perform(context)
    )
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
                output="screen",
            )
        ]
    if head in ("manual", "imu"):
        executable = "manual_head.py" if head == "manual" else "imu_head.py"
        parameters = [str(_share("aha_perception") / "config" / "head.yaml")]
        if head == "imu":
            parameters.append({"mode": "state"})
        return [
            Node(
                package="aha_perception",
                executable=executable,
                parameters=parameters,
                output="screen",
            )
        ]
    return []


def _bridges(context):
    """aha_manipulation's arm trajectory bridges, as arms.launch.py starts
    them (same node names and config/arm.yaml)."""

    def on(name):
        return LaunchConfiguration(name).perform(context).lower() in ("true", "1")

    if not (on("arms") and on("trajectory_bridge")):
        return []
    script = (
        Path(get_package_prefix("aha_manipulation"))
        / "lib"
        / "aha_manipulation"
        / "arm_trajectory_bridge.py"
    )
    config = _share("aha_manipulation") / "config" / "arm.yaml"
    return [
        ExecuteProcess(
            cmd=[
                "python3",
                str(script),
                "--ros-args",
                "-r",
                f"__node:=aha_{side}_arm_bridge",
                "--params-file",
                str(config),
            ],
            name=f"aha_{side}_arm_bridge",
            output="screen",
            emulate_tty=True,
        )
        for side in ("left", "right")
        if on(f"use_{side}")
    ]


def _camera(context):
    if LaunchConfiguration("camera").perform(context).lower() not in ("true", "1"):
        return []
    imu = LaunchConfiguration("head").perform(context) == "imu"
    return [
        IncludeLaunchDescription(
            _launch("aha_perception", "realsense.launch.py"),
            # The IMU's HID stream is opened only when the head tilt uses it.
            launch_arguments={"imu": "true" if imu else "false"}.items(),
        )
    ]


def generate_launch_description():
    odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="odom_tf",
        arguments=["--frame-id", "odom", "--child-frame-id", "base_footprint"],
        output="screen",
    )
    joint_states = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{"source_list": [HEAD_STATES, *ARM_STATES], "rate": 30}],
        output="screen",
    )
    # Scoped: the include's trajectory_bridge:=false must not replace this
    # launch's own (the bridges are started by _bridges instead).
    arms = GroupAction(
        [
            IncludeLaunchDescription(
                _launch("aha_bringup", "arms.launch.py"),
                launch_arguments={
                    "left_device": LaunchConfiguration("left_device"),
                    "right_device": LaunchConfiguration("right_device"),
                    "use_left": LaunchConfiguration("use_left"),
                    "use_right": LaunchConfiguration("use_right"),
                    "trajectory_bridge": "false",
                    "record": LaunchConfiguration("record"),
                }.items(),
            )
        ],
        scoped=True,
        condition=IfCondition(LaunchConfiguration("arms")),
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
                    _share("aha_test_frame") / "config" / "frame_table_202610.yaml"
                ),
                description="Frame model (aha_test_frame/frame.py format)",
            ),
            DeclareLaunchArgument("arms", default_value="true"),
            DeclareLaunchArgument("use_left", default_value="true"),
            DeclareLaunchArgument("use_right", default_value="true"),
            DeclareLaunchArgument("left_device", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("right_device", default_value="/dev/ttyUSB2"),
            DeclareLaunchArgument(
                "trajectory_bridge",
                default_value="true",
                description="Arm trajectory bridges (arms.launch.py)",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="false",
                description="rosbag of every topic (arms.launch.py)",
            ),
            DeclareLaunchArgument(
                "head",
                default_value="servo",
                choices=["servo", "manual", "imu", "none"],
                description="servo: head servos on head_port; manual: commanded "
                "angles as the state (manual_head.py); imu: tilt from the D435i "
                "IMU (imu_head.py); none: no head node",
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
            joint_states,
            arms,
            OpaqueFunction(function=_bridges),
            OpaqueFunction(function=_head),
            OpaqueFunction(function=_camera),
            pointcloud,
            planes,
            rviz,
        ]
    )

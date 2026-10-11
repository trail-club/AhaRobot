"""Real head (and camera) test: move the real head from the simulation UI,
without Gazebo.

The head servos are driven by aha_servo/servo_trajectory_bridge.py, which
serves the same /head_controller/joint_trajectory topic and
follow_joint_trajectory action as the simulation's JointTrajectoryController,
so the control panel, head_look_at.py and teleop_head.py work as in the
simulation.

  ros2 launch aha_perception real_head_camera.launch.py
  ros2 launch aha_perception real_head_camera.launch.py camera:=true
  ros2 launch aha_perception real_head_camera.launch.py camera:=true camera_source:=realsense planes:=true
  ros2 launch aha_perception real_head_camera.launch.py port:=/dev/ttyUSB1 baud:=921600 rviz:=false
  ros2 launch aha_perception real_head_camera.launch.py head:=manual camera:=true planes:=true
  ros2 launch aha_perception real_head_camera.launch.py head:=imu camera:=true camera_source:=realsense

Starts:
  - robot_state_publisher (xacro sim:=false, wall clock)
  - joint_state_publisher: /joint_states with every movable joint; the head
    comes from /head_controller/joint_states, all other joints read 0
  - head:=servo (default)  aha_servo servo_trajectory_bridge.py with config
                           (default aha_servo/config/head.yaml; port/baud
                           override it when given); with imu_check:=true also
                           imu_head.py mode:=check (servo vs IMU tilt)
  - head:=manual           manual_head.py: no servos, the commanded angles
                           become the head state (match a hand-tilted camera
                           with the panel sliders)
  - head:=imu              imu_head.py: no servos, tilt from the D435i IMU,
                           pan from the commanded angles
                           (needs camera:=true camera_source:=realsense)
  - rviz:=true (default)  sim_view.launch.py with use_sim_time:=false,
                          fixed_frame:=base_link (RViz, control panel,
                          look-at; the base pad does nothing)
  - camera:=true          pointcloud.launch.py and the camera source:
                            camera_source:=rosbridge (default)
                              rosbridge_server on 127.0.0.1:9090; the D435i
                              stream (stream_realsense.py) is started
                              separately, see aha_perception/README.md
                            camera_source:=realsense
                              realsense.launch.py (color_profile,
                              depth_profile; IMU only for head:=imu or
                              imu_check:=true)
  - planes:=true          plane_detection.launch.py (needs the camera stream,
                          i.e. camera:=true)

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


def flag(context, name):
    return LaunchConfiguration(name).perform(context).lower() in ("true", "1")


def needs_imu(context):
    return LaunchConfiguration("head").perform(context) == "imu" or flag(
        context, "imu_check"
    )


def perception_launch(name):
    return PythonLaunchDescriptionSource(
        PathJoinSubstitution([FindPackageShare("aha_perception"), "launch", name])
    )


def imu_head(mode):
    return Node(
        package="aha_perception",
        executable="imu_head.py",
        parameters=[
            PathJoinSubstitution(
                [FindPackageShare("aha_perception"), "config", "head.yaml"]
            ),
            {"mode": mode},
        ],
        output="screen",
    )


def head_bridge(context):
    head = LaunchConfiguration("head").perform(context)
    if head == "manual":
        head_params = PathJoinSubstitution(
            [FindPackageShare("aha_perception"), "config", "head.yaml"]
        )
        return [
            Node(
                package="aha_perception",
                executable="manual_head.py",
                parameters=[head_params],
                output="screen",
            )
        ]
    if head == "imu":
        return [imu_head("state")]
    overrides = {}
    port = LaunchConfiguration("port").perform(context)
    baud = LaunchConfiguration("baud").perform(context)
    if port:
        overrides["port"] = port
    if baud:
        overrides["baud"] = int(baud)
    actions = [
        Node(
            package="aha_servo",
            executable="servo_trajectory_bridge.py",
            name="aha_head_servo_bridge",
            parameters=[LaunchConfiguration("config").perform(context), overrides],
            output="screen",
        )
    ]
    if flag(context, "imu_check"):
        actions.append(imu_head("check"))
    return actions


def check_arguments(context):
    source = LaunchConfiguration("camera_source").perform(context)
    if needs_imu(context) and not (flag(context, "camera") and source == "realsense"):
        raise RuntimeError(
            "head:=imu and imu_check:=true need camera:=true camera_source:=realsense"
        )
    return []


def camera_source(context):
    if not flag(context, "camera"):
        return []
    if LaunchConfiguration("camera_source").perform(context) == "realsense":
        return [
            IncludeLaunchDescription(
                perception_launch("realsense.launch.py"),
                launch_arguments={
                    "color_profile": LaunchConfiguration("color_profile"),
                    "depth_profile": LaunchConfiguration("depth_profile"),
                    "imu": "true" if needs_imu(context) else "false",
                }.items(),
            )
        ]
    # Host network: bind to loopback so the bridge is not exposed on the LAN.
    return [
        IncludeLaunchDescription(
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
        perception_launch("sim_view.launch.py"),
        launch_arguments={"use_sim_time": "false", "fixed_frame": "base_link"}.items(),
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    pointcloud = IncludeLaunchDescription(
        perception_launch("pointcloud.launch.py"),
        launch_arguments={"use_sim_time": "false"}.items(),
        condition=IfCondition(LaunchConfiguration("camera")),
    )

    planes = IncludeLaunchDescription(
        perception_launch("plane_detection.launch.py"),
        condition=IfCondition(LaunchConfiguration("planes")),
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
                "head",
                default_value="servo",
                choices=["servo", "manual", "imu"],
                description="servo: drive the head servos; manual: no servos, "
                "manual_head.py publishes the commanded angles as the head state; "
                "imu: no servos, imu_head.py publishes the IMU tilt and the "
                "commanded pan (camera_source:=realsense)",
            ),
            DeclareLaunchArgument(
                "imu_check",
                default_value="false",
                description="With head:=servo, warn when the servo tilt and the "
                "IMU tilt differ by more than 3 deg (camera_source:=realsense)",
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
                description="Start the camera source and the point cloud node",
            ),
            DeclareLaunchArgument(
                "camera_source",
                default_value="rosbridge",
                choices=["rosbridge", "realsense"],
                description="rosbridge: rosbridge (127.0.0.1:9090) for "
                "stream_realsense.py; realsense: the RealSense ROS driver "
                "(realsense.launch.py)",
            ),
            DeclareLaunchArgument(
                "color_profile",
                default_value="640,480,15",
                description="camera_source:=realsense color width,height,fps",
            ),
            DeclareLaunchArgument(
                "depth_profile",
                default_value="640,480,15",
                description="camera_source:=realsense depth width,height,fps",
            ),
            DeclareLaunchArgument(
                "planes",
                default_value="false",
                description="Start plane detection on the head depth camera "
                "(with camera:=true)",
            ),
            OpaqueFunction(function=check_arguments),
            rsp,
            jsp,
            OpaqueFunction(function=head_bridge),
            sim_view,
            OpaqueFunction(function=camera_source),
            pointcloud,
            planes,
        ]
    )

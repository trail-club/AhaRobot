"""Run the staged apple-pick simulation and its scripted controller task."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    sim_launch = PathJoinSubstitution(
        [FindPackageShare("aha_bringup"), "launch", "sim.launch.py"]
    )
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_launch),
        launch_arguments={
            "world": "apple_pick.sdf",
            "headless": LaunchConfiguration("headless"),
            "apple_pick_mode": "true",
        }.items(),
    )
    task = Node(
        package="aha_bringup",
        executable="pick_apple.py",
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    world_poses = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/model/apple/pose@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V"
        ],
        output="screen",
    )
    grasp_latch = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/apple/attach@std_msgs/msg/Empty]gz.msgs.Empty"],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Gazebo without its graphical interface",
            ),
            simulator,
            world_poses,
            grasp_latch,
            task,
        ]
    )

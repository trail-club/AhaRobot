"""Fixed-map Nav2 for simulation; do not run alongside slam.launch.py."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    share = FindPackageShare("aha_navigation")
    params = PathJoinSubstitution([share, "config", "nav2.yaml"])
    common = {"use_sim_time": True}
    servers = [
        ("nav2_map_server", "map_server"),
        ("nav2_amcl", "amcl"),
        ("nav2_controller", "controller_server"),
        ("nav2_planner", "planner_server"),
        ("nav2_smoother", "smoother_server"),
        ("nav2_behaviors", "behavior_server"),
        ("nav2_bt_navigator", "bt_navigator"),
    ]
    nodes = []
    for package, name in servers:
        overrides = dict(common)
        if name == "map_server":
            overrides["yaml_filename"] = LaunchConfiguration("map")
        nodes.append(
            Node(
                package=package,
                executable=name,
                name=name,
                parameters=[params, overrides],
                remappings=[("cmd_vel", "/diff_drive_controller/cmd_vel")],
                output="screen",
            )
        )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=PathJoinSubstitution([share, "maps", "rcjo2026.yaml"]),
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="aha_nav_scan_bridge",
                arguments=["/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"],
                parameters=[common],
                output="screen",
            ),
            *nodes,
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                parameters=[
                    common,
                    {"autostart": True, "node_names": [n for _, n in servers]},
                ],
                output="screen",
            ),
        ]
    )

"""Fixed-map Nav2 for simulation; do not run alongside slam.launch.py.

`navigation:=false` starts only the map and AMCL (localization), for a
stand-in NavigateToPose server such as aha_tasks sim_teleport.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

SERVERS = [
    ("nav2_map_server", "map_server"),
    ("nav2_amcl", "amcl"),
    ("nav2_controller", "controller_server"),
    ("nav2_planner", "planner_server"),
    ("nav2_smoother", "smoother_server"),
    ("nav2_behaviors", "behavior_server"),
    ("nav2_bt_navigator", "bt_navigator"),
]
LOCALIZATION = ("map_server", "amcl")
COMMON = {"use_sim_time": True}


def selected_servers(context):
    if LaunchConfiguration("navigation").perform(context) == "true":
        return SERVERS
    return [(package, name) for package, name in SERVERS if name in LOCALIZATION]


def servers(context):
    params = PathJoinSubstitution(
        [FindPackageShare("aha_navigation"), "config", "nav2.yaml"]
    )
    pose = LaunchConfiguration("initial_pose").perform(context)
    nodes = []
    chosen = selected_servers(context)
    for package, name in chosen:
        overrides = dict(COMMON)
        if name == "map_server":
            overrides["yaml_filename"] = LaunchConfiguration("map")
        if name == "amcl" and pose:
            x, y, yaw = (float(v) for v in pose.split(","))
            overrides.update(
                {"initial_pose.x": x, "initial_pose.y": y, "initial_pose.yaw": yaw}
            )
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
    nodes.append(
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            parameters=[
                COMMON,
                {"autostart": True, "node_names": [name for _, name in chosen]},
            ],
            output="screen",
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("aha_navigation"), "maps", "rcjo2026.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "initial_pose",
                default_value="",
                description="AMCL initial pose 'x,y,yaw' in map; empty uses nav2.yaml",
            ),
            DeclareLaunchArgument(
                "navigation",
                default_value="true",
                choices=["true", "false"],
                description="false: only map_server and AMCL, no planner/controller/BT",
            ),
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="aha_nav_scan_bridge",
                arguments=["/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"],
                parameters=[COMMON],
                output="screen",
            ),
            OpaqueFunction(function=servers),
        ]
    )

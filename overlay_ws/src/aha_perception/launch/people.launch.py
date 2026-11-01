"""Track people in the head RGB-D camera with SAM 3.1 on the DGX Spark.

Inputs are the standard head camera topics from the simulation bridge or the
real camera. The DGX is reached over SSH (Cloudflare One / WARP); see the
package README.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# name: (default, parameter type, description)
ARGUMENTS = {
    "target_frame": ("map", str, "Frame of the published people; empty = camera"),
    "ssh_host": ("10.99.0.1", str, "DGX SSH host or an SSH config alias"),
    "ssh_port": ("2222", int, "SSH port; 0 = from the SSH config"),
    "ssh_user": ("", str, "directory-access Unix username; empty = SSH config"),
    "identity_file": ("", str, "SSH private key; empty = default keys or agent"),
    "use_sim_time": ("false", bool, "Use the simulation clock"),
}


def generate_launch_description():
    return LaunchDescription(
        [
            *(
                DeclareLaunchArgument(name, default_value=value, description=text)
                for name, (value, _, text) in ARGUMENTS.items()
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("aha_perception"), "config", "people.yaml"]
                ),
            ),
            Node(
                package="aha_perception",
                executable="person_perception.py",
                name="person_perception",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        name: ParameterValue(LaunchConfiguration(name), value_type=kind)
                        for name, (_, kind, _) in ARGUMENTS.items()
                    },
                ],
                output="screen",
                # Leave time to remove the DGX container after Ctrl-C.
                sigterm_timeout="20",
            ),
        ]
    )

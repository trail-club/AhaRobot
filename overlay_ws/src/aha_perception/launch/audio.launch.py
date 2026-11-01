"""Audio stack: ASR + TTS (+ debug UI). Run inside the aha_audio container.

target chooses where the models run: dgx / trailpc over SSH (WARP), or local in
these processes. See docs/audio.md.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# name: (default, parameter type, description)
INFERENCE_ARGUMENTS = {
    "target": ("dgx", str, "Where the models run: dgx | trailpc | local"),
    "ssh_host": ("", str, "SSH host or config alias; empty = address of the target"),
    "ssh_port": ("2222", int, "SSH port; 0 = from the SSH config"),
    "ssh_user": ("", str, "directory-access Unix username; empty = SSH config"),
    "identity_file": ("", str, "SSH private key; empty = default keys or agent"),
}


def generate_launch_description():
    config = [FindPackageShare("aha_perception"), "config", "audio"]
    params = LaunchConfiguration("params_file")
    models = LaunchConfiguration("models_file")
    language = LaunchConfiguration("language")
    inference = {
        name: ParameterValue(LaunchConfiguration(name), value_type=kind)
        for name, (_, kind, _) in INFERENCE_ARGUMENTS.items()
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution([*config, "params.yaml"]),
            ),
            DeclareLaunchArgument(
                "models_file",
                default_value=PathJoinSubstitution([*config, "models.yaml"]),
            ),
            DeclareLaunchArgument(
                "language", default_value="en", description="ASR language: en | ja"
            ),
            *(
                DeclareLaunchArgument(name, default_value=value, description=text)
                for name, (value, _, text) in INFERENCE_ARGUMENTS.items()
            ),
            DeclareLaunchArgument("use_asr", default_value="true"),
            DeclareLaunchArgument("use_tts", default_value="true"),
            DeclareLaunchArgument("use_debug_ui", default_value="true"),
            Node(
                package="aha_perception",
                executable="audio_asr_node",
                name="aha_audio_asr",
                output="screen",
                parameters=[
                    params,
                    {"models_file": models, "language": language, **inference},
                ],
                # Leave time to remove the worker container after Ctrl-C.
                sigterm_timeout="20",
                condition=IfCondition(LaunchConfiguration("use_asr")),
            ),
            Node(
                package="aha_perception",
                executable="audio_tts_node",
                name="aha_audio_tts",
                output="screen",
                parameters=[params, {"models_file": models, **inference}],
                sigterm_timeout="20",
                condition=IfCondition(LaunchConfiguration("use_tts")),
            ),
            Node(
                package="aha_perception",
                executable="audio_debug_ui",
                name="aha_audio_debug_ui",
                output="screen",
                parameters=[params],
                condition=IfCondition(LaunchConfiguration("use_debug_ui")),
            ),
        ]
    )

"""Audio stack: ASR + TTS (+ debug UI). Run inside the aha_audio container."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = [FindPackageShare('aha_perception'), 'config', 'audio']
    params = LaunchConfiguration('params_file')
    models = LaunchConfiguration('models_file')
    language = LaunchConfiguration('language')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=PathJoinSubstitution([*config, 'params.yaml'])),
        DeclareLaunchArgument('models_file', default_value=PathJoinSubstitution([*config, 'models.yaml'])),
        DeclareLaunchArgument('language', default_value='en', description='ASR language: en | ja'),
        DeclareLaunchArgument('use_asr', default_value='true'),
        DeclareLaunchArgument('use_tts', default_value='true'),
        DeclareLaunchArgument('use_debug_ui', default_value='true'),
        Node(package='aha_perception', executable='audio_asr_node', name='aha_audio_asr',
             output='screen', parameters=[params, {'models_file': models, 'language': language}],
             condition=IfCondition(LaunchConfiguration('use_asr'))),
        Node(package='aha_perception', executable='audio_tts_node', name='aha_audio_tts',
             output='screen', parameters=[params, {'models_file': models}],
             condition=IfCondition(LaunchConfiguration('use_tts'))),
        Node(package='aha_perception', executable='audio_debug_ui', name='aha_audio_debug_ui',
             output='screen', parameters=[params],
             condition=IfCondition(LaunchConfiguration('use_debug_ui'))),
    ])

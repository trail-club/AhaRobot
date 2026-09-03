"""Perception launch stub.

Owner: perception squad. Currently empty — camera bridges and detection
nodes will be added here.
"""
from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="[aha_perception] stub launch — replace with camera + detection pipeline"),
    ])

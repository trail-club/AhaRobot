"""Manipulation launch stub.

Owner: manipulation squad. Currently empty — MoveIt move_group / pick server
launches will be added here.
"""
from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="[aha_manipulation] stub launch — replace with MoveIt / pick_server setup"),
    ])

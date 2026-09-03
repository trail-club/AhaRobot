"""Navigation launch stub.

Owner: navigation squad. Currently empty — real Nav2 / slam_toolbox launches
will be added here. Kept as a valid launch file so aha_bringup can include
it with `use_nav:=true` without erroring.
"""
from launch import LaunchDescription
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        LogInfo(msg="[aha_navigation] stub launch — replace with Nav2 / slam_toolbox setup"),
    ])

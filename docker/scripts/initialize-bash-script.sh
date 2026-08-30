#!/bin/bash
# Sourced from ~/.bashrc inside the container.

# ROS 2 Jazzy
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
fi

# Overlay workspace (built at /app/overlay_ws)
if [ -f /app/overlay_ws/install/setup.bash ]; then
    source /app/overlay_ws/install/setup.bash
fi

# Convenience aliases
alias aha_build='cd /app/overlay_ws && colcon build --symlink-install --packages-up-to aha_bringup'
alias aha_sim='ros2 launch aha_bringup sim.launch.py'
alias aha_view='ros2 launch aha_description view_robot.launch.py'
alias aha_teleop='ros2 run aha_bringup teleop_base.sh'
alias aha_demo='ros2 run aha_bringup demo_arms.sh'

# Prompt tag so users notice they are in the container
export PS1="\[\e[1;35m\][aha]\[\e[0m\] \u@\h:\w$ "

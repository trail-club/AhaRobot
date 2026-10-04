cmake_minimum_required(VERSION 3.8)
project(aha_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_perception.head_config: head.yaml loader shared by the scripts below
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/head_look_at.py scripts/sim_control_panel.py scripts/teleop_head.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

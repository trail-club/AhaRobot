cmake_minimum_required(VERSION 3.8)
project(aha_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/teleop_base.sh scripts/demo_arms.sh
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

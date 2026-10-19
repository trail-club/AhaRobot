cmake_minimum_required(VERSION 3.8)
project(aha_test_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)

# Interfaces of the test-only packages (tests/ros), not of the robot.
rosidl_generate_interfaces(${PROJECT_NAME}
  action/PlaceObject.action
  DEPENDENCIES geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

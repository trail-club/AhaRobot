cmake_minimum_required(VERSION 3.8)
project(aha_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Cross-team interfaces. Add here as squads land real definitions.
# Convention: msg = one-shot data, srv = request/reply, action = long-running goal.
rosidl_generate_interfaces(${PROJECT_NAME}
  msg/DetectedObject.msg
  msg/DetectedObjectArray.msg
  srv/SetNavGoal.srv
  action/PickObject.action
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

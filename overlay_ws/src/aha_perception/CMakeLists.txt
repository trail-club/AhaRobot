cmake_minimum_required(VERSION 3.8)
project(aha_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_perception.head_config: head.yaml loader shared by the scripts below
# aha_perception.planes: plane detection candidates
# aha_perception.plane_detection_io: ROS message conversions for plane_detection_node.py
# aha_perception.tf_jump: TF listener that detects sim restarts
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/head_look_at.py
    scripts/plane_detection_node.py
    scripts/sim_control_panel.py
    scripts/teleop_head.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_sim_control_panel test/test_sim_control_panel.py TIMEOUT 60)
  foreach(test_name classify detectors merge normals preprocess)
    ament_add_pytest_test(test_planes_${test_name} test/test_planes_${test_name}.py TIMEOUT 120)
  endforeach()
  ament_add_pytest_test(test_plane_detection_io test/test_plane_detection_io.py TIMEOUT 120)
endif()

ament_package()

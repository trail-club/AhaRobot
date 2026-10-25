cmake_minimum_required(VERSION 3.8)
project(aha_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_perception: head.yaml loader and the person perception core shared by the scripts below
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/head_look_at.py
    scripts/person_perception.py
    scripts/sim_control_panel.py
    scripts/teleop_head.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_sim_control_panel test/test_sim_control_panel.py TIMEOUT 60)
  ament_add_pytest_test(test_person_perception test/test_person_perception.py TIMEOUT 120)
endif()

ament_package()

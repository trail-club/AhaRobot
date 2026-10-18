cmake_minimum_required(VERSION 3.8)
project(aha_manipulation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config srdf rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/joint_state_bounds.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_moveit_configuration test/test_configuration.py)
  ament_add_pytest_test(test_manipulation_scripts test/test_scripts.py)
endif()

ament_package()

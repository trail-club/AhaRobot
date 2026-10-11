cmake_minimum_required(VERSION 3.8)
project(aha_navigation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config maps rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_demo_map test/test_demo_map.py)
endif()
ament_package()

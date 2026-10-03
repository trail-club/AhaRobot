cmake_minimum_required(VERSION 3.8)
project(aha_gazebo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY worlds launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_gazebo_world test/test_world.py)
  ament_add_pytest_test(test_gazebo_launch test/test_launch.py)
endif()
ament_package()

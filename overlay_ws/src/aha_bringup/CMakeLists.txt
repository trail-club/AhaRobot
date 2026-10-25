cmake_minimum_required(VERSION 3.8)
project(aha_bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch scripts
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/teleop_base.sh scripts/teleop_arms.sh scripts/demo_arms.sh
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_real_launch test/test_real_launch.py TIMEOUT 60)
endif()

ament_package()

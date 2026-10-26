cmake_minimum_required(VERSION 3.8)
project(aha_test_ui)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_test_ui.records: status / chat / event records and their HTML (ROS free)
# aha_test_ui.publisher: UiPublisher for modules (ROS)
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/robot_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_records test/test_records.py TIMEOUT 60)
endif()

ament_package()

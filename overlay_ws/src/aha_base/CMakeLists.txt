cmake_minimum_required(VERSION 3.8)
project(aha_base)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_base.diff_drive and aha_base.odrive_usb (ROS-independent)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/base_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_diff_drive test/test_diff_drive.py TIMEOUT 60)
  ament_add_pytest_test(test_odrive_usb test/test_odrive_usb.py TIMEOUT 60)
  ament_add_pytest_test(test_base_node test/test_base_node.py TIMEOUT 240)
endif()

ament_package()

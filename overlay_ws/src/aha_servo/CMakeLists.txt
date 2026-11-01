cmake_minimum_required(VERSION 3.8)
project(aha_servo)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_servo.sts (STS protocol + stdlib serial) and aha_servo.joint_map
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/servo_trajectory_bridge.py scripts/servo_calibrate.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_sts test/test_sts.py TIMEOUT 60)
  ament_add_pytest_test(test_joint_map test/test_joint_map.py TIMEOUT 60)
  ament_add_pytest_test(test_calibrate test/test_calibrate.py TIMEOUT 60)
  ament_add_pytest_test(test_bridge test/test_bridge.py TIMEOUT 180)
endif()

ament_package()

cmake_minimum_required(VERSION 3.8)
project(aha_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_manipulation Python package (arm_map, arm_plan)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config srdf rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/arm_trajectory_bridge.py
    scripts/eef_goal_monitor.py
    scripts/joint_state_bounds.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_moveit_configuration test/test_configuration.py)
  ament_add_pytest_test(test_manipulation_scripts test/test_scripts.py)
  ament_add_pytest_test(test_arm_plan test/test_arm_plan.py TIMEOUT 60)
  ament_add_pytest_test(test_arm_bridge test/test_arm_bridge.py TIMEOUT 180)
endif()

ament_package()

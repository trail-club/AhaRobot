cmake_minimum_required(VERSION 3.8)
project(aha_test_manipulation)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_test_manipulation.place_geometry: object-in-hand convention and place poses (ROS free)
# aha_test_manipulation.scene_objects: planning scene objects from planes and the held object
# aha_test_manipulation.moveit_requests: MoveGroup goals and SRDF named states
# aha_test_manipulation.sim_held_object: Gazebo held object (sim only)
# aha_test_manipulation.frame_moveit: SRDF, limits and controllers of the real robot's frame model
# aha_test_manipulation.gripper_ramp: ramp sequences and contact detection for gripper_handover.py (ROS free)
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config srdf rviz
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES .setup_assistant
  DESTINATION share/${PROJECT_NAME}
)
install(
  PROGRAMS
    scripts/place_server.py
    scripts/sim_hold_object.py
    scripts/gripper_handover.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_moveit_config test/test_moveit_config.py)
  ament_add_pytest_test(test_frame_moveit test/test_frame_moveit.py)
  ament_add_pytest_test(test_place_geometry test/test_place_geometry.py)
  ament_add_pytest_test(test_scene_objects test/test_scene_objects.py)
  ament_add_pytest_test(test_place_selection test/test_place_selection.py)
  ament_add_pytest_test(test_place_markers test/test_place_markers.py)
  ament_add_pytest_test(test_sim_held_object test/test_sim_held_object.py)
  ament_add_pytest_test(test_self_filter test/test_self_filter.py)
  ament_add_pytest_test(test_sim_trials test/test_sim_trials.py)
  ament_add_pytest_test(test_gripper_ramp test/test_gripper_ramp.py)
endif()

ament_package()

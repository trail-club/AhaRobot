cmake_minimum_required(VERSION 3.8)
project(aha_test_frame)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(astra_description REQUIRED)

# aha_test_frame.frame: frame model of the real robot on its table frame (URDF rewrite)
# aha_test_frame.plane_calibration / camera_arm_calibration: head camera mount calibration
# aha_test_frame.arm_cloud_calibration: arm mount and joint offsets from the point cloud
ament_python_install_package(${PROJECT_NAME})

# Head base (printed HeadYaw part with the pan servo) cut out of the upstream
# body mesh for the frame model, relative to the joint_head_pan origin.
set(UPSTREAM_URDF ${astra_description_DIR}/../urdf/astra_description.urdf)
set(UPSTREAM_BODY_MESH ${astra_description_DIR}/../meshes/base_link.STL)
set(HEAD_BASE_MESH ${CMAKE_CURRENT_BINARY_DIR}/meshes/head_base.stl)
add_custom_command(
  OUTPUT ${HEAD_BASE_MESH}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/crop_head_base.py
          ${UPSTREAM_URDF} ${UPSTREAM_BODY_MESH} ${HEAD_BASE_MESH}
  DEPENDS ${UPSTREAM_URDF} ${UPSTREAM_BODY_MESH}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/crop_head_base.py
  COMMENT "Cutting the head base out of the upstream body mesh"
)
add_custom_target(head_base_mesh ALL DEPENDS ${HEAD_BASE_MESH})

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES ${HEAD_BASE_MESH}
  DESTINATION share/${PROJECT_NAME}/meshes
)
install(
  PROGRAMS
    scripts/arm_cloud_calibrate.py
    scripts/camera_arm_calibrate.py
    scripts/camera_plane_calibrate.py
    scripts/frame_calibrate.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_frame test/test_frame.py TIMEOUT 60)
  ament_add_pytest_test(test_frame_calibrate test/test_frame_calibrate.py TIMEOUT 60)
  ament_add_pytest_test(test_real_launch test/test_real_launch.py TIMEOUT 60)
  ament_add_pytest_test(test_plane_calibration test/test_plane_calibration.py TIMEOUT 60)
  ament_add_pytest_test(test_camera_arm_calibration test/test_camera_arm_calibration.py TIMEOUT 60)
  ament_add_pytest_test(test_arm_cloud_calibration test/test_arm_cloud_calibration.py TIMEOUT 300)
  ament_add_pytest_test(test_launch_composition test/test_launch_composition.py TIMEOUT 120)
  ament_add_pytest_test(test_head_base_mesh test/test_head_base_mesh.py TIMEOUT 60)
  ament_add_pytest_test(test_frame_script_help test/test_frame_script_help.py TIMEOUT 120)
endif()

ament_package()

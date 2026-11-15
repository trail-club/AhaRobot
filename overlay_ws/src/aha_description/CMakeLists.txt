cmake_minimum_required(VERSION 3.8)
project(aha_description)

find_package(ament_cmake REQUIRED)
find_package(astra_description REQUIRED)

# Upstream URDF has <limit effort="0" velocity="0"> on every joint, which makes
# gz_ros2_control apply zero torque and the joints stay put. Patch those to
# sensible defaults at build time, and swap the mesh/inertial of the camera
# mount links: link_head_tilt to the D435i bracket
# (meshes/head_cam_mount_d435i.stl, from aha_perception/hardware/head_cam_mount_d435i)
# and link_l6/link_r6 to the gripper base with the D435i wrist part
# (meshes/wrist_cam_mount_d435i_link6.stl, from
# aha_perception/hardware/wrist_cam_mount_d435i). Write to the install tree
# so the top-level xacro can xacro:include the patched copy instead of the raw
# upstream one.
set(UPSTREAM_URDF ${astra_description_DIR}/../urdf/astra_description.urdf)
set(PATCHED_URDF  ${CMAKE_CURRENT_BINARY_DIR}/urdf/astra_description_patched.urdf)
add_custom_command(
  OUTPUT ${PATCHED_URDF}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/urdf
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_limits.py
          ${UPSTREAM_URDF} ${PATCHED_URDF}
  DEPENDS ${UPSTREAM_URDF} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_limits.py
          ${CMAKE_CURRENT_SOURCE_DIR}/meshes/head_cam_mount_d435i.stl
          ${CMAKE_CURRENT_SOURCE_DIR}/meshes/wrist_cam_mount_d435i_link6.stl
  COMMENT "Patching upstream URDF (joint limits, camera mount meshes)"
)
add_custom_target(patched_urdf ALL DEPENDS ${PATCHED_URDF})

install(
  DIRECTORY urdf config launch meshes
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES ${PATCHED_URDF}
  DESTINATION share/${PROJECT_NAME}/urdf
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_patch_limits test/test_patch_limits.py TIMEOUT 60)
  ament_add_pytest_test(test_sensors test/test_sensors.py TIMEOUT 120)
endif()

ament_package()

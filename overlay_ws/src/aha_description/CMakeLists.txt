cmake_minimum_required(VERSION 3.8)
project(aha_description)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(astra_description REQUIRED)

# Upstream URDF has <limit effort="0" velocity="0"> on every joint, which makes
# gz_ros2_control apply zero torque and the joints stay put. Patch those to
# sensible defaults at build time, and swap link_head_tilt's mesh/inertial to
# the D435i bracket (meshes/head_cam_mount_d435i.stl, exported from
# aha_perception/hardware/head_cam_mount_d435i). Write to the install tree
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
  COMMENT "Patching upstream URDF (joint limits, head tilt mesh)"
)
add_custom_target(patched_urdf ALL DEPENDS ${PATCHED_URDF})

# Head base (printed HeadYaw part with the pan servo) cut out of the upstream
# body mesh for the frame model (aha_description/frame.py), relative to the
# joint_head_pan origin.
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
  DIRECTORY urdf config launch meshes
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES ${PATCHED_URDF}
  DESTINATION share/${PROJECT_NAME}/urdf
)
install(
  FILES ${HEAD_BASE_MESH}
  DESTINATION share/${PROJECT_NAME}/meshes
)

# Python library (aha_description.frame) for launch files.
ament_python_install_package(${PROJECT_NAME})
install(
  PROGRAMS scripts/frame_calibrate.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_frame test/test_frame.py TIMEOUT 60)
  ament_add_pytest_test(test_frame_calibrate test/test_frame_calibrate.py TIMEOUT 60)
endif()

ament_package()

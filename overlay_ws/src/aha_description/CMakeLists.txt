cmake_minimum_required(VERSION 3.8)
project(aha_description)

find_package(ament_cmake REQUIRED)
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

# One mesh of the printed display mount parts in display_link coordinates,
# built from the print STLs in hardware/display_mount (see urdf/display.xacro).
set(DISPLAY_MOUNT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware/display_mount)
set(DISPLAY_MOUNT_PARTS
  ${DISPLAY_MOUNT_DIR}/01_frame_adapter.stl
  ${DISPLAY_MOUNT_DIR}/02_lower_left.stl
  ${DISPLAY_MOUNT_DIR}/03_lower_right.stl
  ${DISPLAY_MOUNT_DIR}/04_top_keeper.stl)
set(DISPLAY_MOUNT_MESH ${CMAKE_CURRENT_BINARY_DIR}/meshes/display_mount.stl)
add_custom_command(
  OUTPUT ${DISPLAY_MOUNT_MESH}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/display_mount_mesh.py
          ${DISPLAY_MOUNT_DIR} ${DISPLAY_MOUNT_MESH}
  DEPENDS ${DISPLAY_MOUNT_PARTS} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/display_mount_mesh.py
  COMMENT "Merging the display mount parts into one URDF mesh"
)
add_custom_target(display_mount_mesh ALL DEPENDS ${DISPLAY_MOUNT_MESH})

install(
  DIRECTORY urdf config launch meshes
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES ${PATCHED_URDF}
  DESTINATION share/${PROJECT_NAME}/urdf
)
install(
  FILES ${DISPLAY_MOUNT_MESH}
  DESTINATION share/${PROJECT_NAME}/meshes
)

ament_package()

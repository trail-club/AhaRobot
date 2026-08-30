cmake_minimum_required(VERSION 3.8)
project(aha_description)

find_package(ament_cmake REQUIRED)
find_package(astra_description REQUIRED)

# Upstream URDF has <limit effort="0" velocity="0"> on every joint, which makes
# gz_ros2_control apply zero torque and the joints stay put. Patch those to
# sensible defaults at build time; write to the install tree so the top-level
# xacro can xacro:include the patched copy instead of the raw upstream one.
set(UPSTREAM_URDF ${astra_description_DIR}/../urdf/astra_description.urdf)
set(PATCHED_URDF  ${CMAKE_CURRENT_BINARY_DIR}/urdf/astra_description_patched.urdf)
add_custom_command(
  OUTPUT ${PATCHED_URDF}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/urdf
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_limits.py
          ${UPSTREAM_URDF} ${PATCHED_URDF}
  DEPENDS ${UPSTREAM_URDF} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_limits.py
  COMMENT "Patching upstream URDF joint limits"
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

ament_package()

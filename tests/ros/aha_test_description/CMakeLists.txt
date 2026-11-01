cmake_minimum_required(VERSION 3.8)
project(aha_test_description)

find_package(ament_cmake REQUIRED)
find_package(aha_description REQUIRED)

# Test only: aha_description's controllers.yaml with config/controllers_overrides.yaml.
set(BASE_CONTROLLERS ${aha_description_DIR}/../config/controllers.yaml)
set(OVERRIDES ${CMAKE_CURRENT_SOURCE_DIR}/config/controllers_overrides.yaml)
set(TEST_CONTROLLERS ${CMAKE_CURRENT_BINARY_DIR}/config/controllers.yaml)
add_custom_command(
  OUTPUT ${TEST_CONTROLLERS}
  COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/merge_controllers.py
          ${BASE_CONTROLLERS} ${OVERRIDES} ${TEST_CONTROLLERS}
  DEPENDS ${BASE_CONTROLLERS} ${OVERRIDES}
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/merge_controllers.py
  COMMENT "Merging the test bed controller overrides"
)
add_custom_target(test_controllers ALL DEPENDS ${TEST_CONTROLLERS})

install(DIRECTORY urdf DESTINATION share/${PROJECT_NAME})
install(FILES ${TEST_CONTROLLERS} DESTINATION share/${PROJECT_NAME}/config)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_description test/test_description.py)
endif()

ament_package()

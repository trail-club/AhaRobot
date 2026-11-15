cmake_minimum_required(VERSION 3.8)
project(aha_sim_tasks)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz_rendering_vendor REQUIRED)
find_package(gz_ogre_next_vendor REQUIRED)
find_package(gz-rendering8 REQUIRED COMPONENTS ogre2)
find_package(PkgConfig REQUIRED)
# gz-rendering's ogre2 target does not export the Ogre-Next headers it wraps.
pkg_check_modules(OGRE_NEXT REQUIRED IMPORTED_TARGET OGRE-Next OGRE-Next-Hlms)
pkg_get_variable(OGRE_NEXT_ROOT_INCLUDE OGRE-Next includedir)
find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)

ament_python_install_package(${PROJECT_NAME})
add_library(aha_task_evaluation SHARED src/evaluation_system.cpp)
target_compile_features(aha_task_evaluation PUBLIC cxx_std_17)
target_link_libraries(aha_task_evaluation gz-sim8::gz-sim8 gz-plugin2::register nlohmann_json::nlohmann_json)
install(TARGETS aha_task_evaluation LIBRARY DESTINATION lib)

# Garment simulation shared by the laundry system and its tests.
add_library(aha_laundry_cloth STATIC src/laundry/cloth.cc src/laundry/fold_metrics.cc)
target_compile_features(aha_laundry_cloth PUBLIC cxx_std_17)
target_compile_options(aha_laundry_cloth PRIVATE -O3)
set_target_properties(aha_laundry_cloth PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(aha_laundry_cloth PUBLIC src)
target_link_libraries(aha_laundry_cloth PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

add_library(aha_laundry SHARED src/laundry/laundry_system.cc src/laundry/cloth_renderer.cc)
target_compile_features(aha_laundry PUBLIC cxx_std_17)
target_compile_options(aha_laundry PRIVATE -O2)
target_include_directories(aha_laundry PRIVATE
  ${OGRE_NEXT_ROOT_INCLUDE}/OGRE-Next/Hlms/Pbs ${OGRE_NEXT_ROOT_INCLUDE}/OGRE-Next/Hlms/Common)
target_link_libraries(aha_laundry aha_laundry_cloth gz-sim8::gz-sim8 gz-plugin2::register
  gz-rendering8::gz-rendering8-ogre2 PkgConfig::OGRE_NEXT)
install(TARGETS aha_laundry LIBRARY DESTINATION lib)
install(DIRECTORY launch worlds config DESTINATION share/${PROJECT_NAME})

# Laundry-room meshes, textures and T-shirt patterns are generated, not stored.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(LAUNDRY_ASSETS ${CMAKE_CURRENT_BINARY_DIR}/laundry)
add_custom_command(
  OUTPUT ${LAUNDRY_ASSETS}/catalog.json
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/aha_sim_tasks/laundry_assets.py
    --output ${LAUNDRY_ASSETS}
  DEPENDS aha_sim_tasks/laundry_assets.py aha_sim_tasks/laundry_pattern.py
  COMMENT "Generating laundry room meshes, textures and garment patterns")
add_custom_target(laundry_assets ALL DEPENDS ${LAUNDRY_ASSETS}/catalog.json)
install(DIRECTORY ${LAUNDRY_ASSETS} DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/evaluate DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_gtest REQUIRED)
  add_executable(ycb_physics_probe test/ycb_physics_probe.cpp)
  target_compile_features(ycb_physics_probe PRIVATE cxx_std_17)
  target_link_libraries(ycb_physics_probe gz-sim8::gz-sim8 nlohmann_json::nlohmann_json)
  ament_add_gtest(test_cloth test/test_cloth.cpp TIMEOUT 120)
  target_link_libraries(test_cloth aha_laundry_cloth)
  target_compile_definitions(test_cloth PRIVATE
    TSHIRT_PATTERN="${LAUNDRY_ASSETS}/patterns/tshirt_M.json")
  add_dependencies(test_cloth laundry_assets)
  ament_add_pytest_test(test_task_laundry test/test_laundry.py TIMEOUT 120)
  ament_add_pytest_test(test_task_physics test/test_physics.py TIMEOUT 180
    ENV "YCB_PHYSICS_PROBE=$<TARGET_FILE:ycb_physics_probe>")
  ament_add_gtest(test_finger_contacts test/test_finger_contacts.cpp)
  target_include_directories(test_finger_contacts PRIVATE src)
  target_link_libraries(test_finger_contacts gz-sim8::gz-sim8)
  ament_add_gtest(test_passive_evaluation test/test_passive_evaluation.cpp)
  target_link_libraries(test_passive_evaluation gz-sim8::gz-sim8 nlohmann_json::nlohmann_json)
  target_compile_definitions(test_passive_evaluation PRIVATE
    EVALUATION_PLUGIN_PATH="$<TARGET_FILE:aha_task_evaluation>")
  add_dependencies(test_passive_evaluation aha_task_evaluation)
  ament_add_pytest_test(test_task_evaluation test/test_evaluation.py)
  ament_add_pytest_test(test_task_scene test/test_scene.py)
  ament_add_pytest_test(test_task_sensor_noise test/test_sensor_noise.py)
  ament_add_pytest_test(test_task_policy test/test_policy.py)
  ament_add_pytest_test(test_task_moveit test/test_moveit.py)
  ament_add_pytest_test(test_task_policy_process test/test_policy_process.py TIMEOUT 60)
  ament_add_pytest_test(test_task_ros_adapter test/test_ros_adapter.py)
  ament_add_pytest_test(test_task_runner test/test_runner.py)
  ament_add_pytest_test(test_task_codex_policy test/test_codex_policy.py)
  ament_add_pytest_test(test_task_stepping test/test_stepping.py)
  ament_add_pytest_test(test_task_video test/test_video.py)
endif()
ament_package()

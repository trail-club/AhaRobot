cmake_minimum_required(VERSION 3.8)
project(aha_test_orchestration)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_test_orchestration.llm: OpenAI-compatible chat client and validated function calls
# aha_test_orchestration.scene: objects on tracked surfaces and furniture levels (ROS free)
# aha_test_orchestration.place_decision: LLM inputs / answers of the place variants (ROS free)
# aha_test_orchestration.place_runner: observe, ask the LLM and send PlaceObject (ROS)
# aha_test_orchestration.base_motion: short base moves on the odometry (ROS)
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/place_with_llm.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_scene test/test_scene.py TIMEOUT 60)
  ament_add_pytest_test(test_place_decision test/test_place_decision.py TIMEOUT 60)
  ament_add_pytest_test(test_llm test/test_llm.py TIMEOUT 60)
  ament_add_pytest_test(test_place_runner test/test_place_runner.py TIMEOUT 60)
  ament_add_pytest_test(test_base_motion test/test_base_motion.py TIMEOUT 60)
  ament_add_pytest_test(test_viewpoint test/test_viewpoint.py TIMEOUT 60)
endif()

ament_package()

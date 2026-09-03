cmake_minimum_required(VERSION 3.8)
project(aha_manipulation)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config srdf rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

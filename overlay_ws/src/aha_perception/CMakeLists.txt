cmake_minimum_required(VERSION 3.8)
project(aha_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Python package (aha_perception.audio = ASR / TTS stack). web/ も一緒に install される。
ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    scripts/audio_asr_node
    scripts/audio_tts_node
    scripts/audio_debug_ui
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL
)

ament_package()

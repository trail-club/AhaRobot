cmake_minimum_required(VERSION 3.8)
project(aha_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# aha_perception.head_config: head.yaml loader shared by the scripts below
# aha_perception.head_follow: commanded head position for manual_head.py / imu_head.py
# aha_perception.plane_calibration / camera_arm_calibration: head camera mount calibration
# aha_perception.arm_cloud_calibration: arm mount yaw and joint offsets from the point cloud
# aha_perception.imu_tilt: head camera tilt/roll from the D435i accelerometer
# aha_perception.planes: plane detection candidates
# aha_perception.place: place candidates on the detected support planes
# aha_perception.surfaces: surfaces with stable IDs over the plane detections
# aha_perception.furniture: furniture config and surface-to-level association
# aha_perception.plane_detection_io: ROS message conversions for the depth nodes
# aha_perception.depth_noise: D435i depth noise model for sim_depth_noise.py
# aha_perception.tf_jump: TF listener that detects sim restarts
# aha_perception.object_list: object labels, categories, sizes and sim model names
# aha_perception.sim_object_detection: visibility and corruption of simulated detections
# aha_perception.people, aha_perception.dgx: person perception core and DGX worker client for person_perception.py
ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/arm_cloud_calibrate.py
    scripts/camera_arm_calibrate.py
    scripts/camera_plane_calibrate.py
    scripts/head_look_at.py
    scripts/imu_head.py
    scripts/manual_head.py
    scripts/person_perception.py
    scripts/plane_detection_node.py
    scripts/sim_control_panel.py
    scripts/sim_depth_noise.py
    scripts/sim_object_detector.py
    scripts/teleop_head.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_sim_control_panel test/test_sim_control_panel.py TIMEOUT 60)
  foreach(test_name classify detectors merge normals preprocess regression)
    ament_add_pytest_test(test_planes_${test_name} test/test_planes_${test_name}.py TIMEOUT 120)
  endforeach()
  ament_add_pytest_test(test_plane_detection_io test/test_plane_detection_io.py TIMEOUT 120)
  ament_add_pytest_test(test_depth_noise test/test_depth_noise.py TIMEOUT 120)
  ament_add_pytest_test(test_place test/test_place.py TIMEOUT 120)
  ament_add_pytest_test(test_plane_calibration test/test_plane_calibration.py TIMEOUT 60)
  ament_add_pytest_test(test_camera_arm_calibration test/test_camera_arm_calibration.py TIMEOUT 60)
  ament_add_pytest_test(test_arm_cloud_calibration test/test_arm_cloud_calibration.py TIMEOUT 300)
  ament_add_pytest_test(test_surfaces test/test_surfaces.py TIMEOUT 120)
  ament_add_pytest_test(test_furniture test/test_furniture.py TIMEOUT 60)
  ament_add_pytest_test(test_plane_detection_node test/test_plane_detection_node.py TIMEOUT 60)
  ament_add_pytest_test(test_plane_detection_service test/test_plane_detection_service.py TIMEOUT 120)
  ament_add_pytest_test(test_imu_tilt test/test_imu_tilt.py TIMEOUT 60)
  ament_add_pytest_test(test_object_list test/test_object_list.py TIMEOUT 60)
  ament_add_pytest_test(test_sim_object_detection test/test_sim_object_detection.py TIMEOUT 60)
  ament_add_pytest_test(test_person_perception test/test_person_perception.py TIMEOUT 120)
endif()

ament_package()

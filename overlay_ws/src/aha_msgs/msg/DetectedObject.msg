# A single object detected by the perception pipeline.
# Owner: perception squad. Consumers: manipulation, navigation.

std_msgs/Header header          # frame_id = camera_optical_frame at detection time
string       class_name         # e.g. "bottle", "person"
int32        track_id           # -1 if untracked
float32      confidence         # [0.0, 1.0]

# Pose of the object center in header.frame_id.
geometry_msgs/PoseWithCovariance pose

# Optional axis-aligned bbox in the object's own frame (meters).
geometry_msgs/Vector3 size

# A person tracked by aha_perception person_perception. Owner: perception squad.
# Consumers: tasks (find / memorize / locate guests), gaze, person following.

uint32 id                          # track ID; a new ID after the person is lost
string name                        # remembered name; empty until remembered or matched by the VLM
float32 score                      # SAM 3.1 detection score of the last sighting
geometry_msgs/Point position       # body centre (median of the mask's depth points)
geometry_msgs/Point head           # top of the body (median of the mask's upper points)
builtin_interfaces/Time last_seen  # stamp of the image of the last sighting
bool visible                       # seen in the image of this message
int32[4] bbox                      # last sighting in the image, [x0, y0, x1, y1] pixels

# From depth and body keypoints only, and only when clear; empty otherwise.
# The ClassifyPerson service also asks the VLM.
string posture                     # standing / sitting / lying
string gesture                     # waving / raising_left_arm / raising_right_arm / raising_both_arms /
                                   # pointing_left / pointing_right / none (left / right: the person's own)

# 17 COCO body keypoints of the last sighting, in this order: nose, left_eye, right_eye,
# left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist,
# right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.
# Positions are in the frame of the message; empty arrays when keypoints are off.
geometry_msgs/Point[] keypoints
float32[] keypoint_scores          # 0: not seen, so the position is meaningless

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

# Tracked surfaces after one plane detection. Owner: perception squad.
# Consumers: task / manipulation.

std_msgs/Header header          # stamp of the depth image, frame_id = tracking frame (odom)
Surface[]    surfaces           # sorted by id; includes surfaces out of view (visible false)

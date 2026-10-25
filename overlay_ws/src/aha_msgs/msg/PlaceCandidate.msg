# A free spot on a support plane where an object of the requested size fits.
# Owner: perception squad. Consumers: manipulation.

# Center of the object footprint on the surface, in base_footprint at the
# detection stamp (header.stamp): transform it with that stamp, the robot
# may have moved since. Orientation: z along the surface normal, x in the
# surface heading the yaw (atan2 of its x y in header.frame_id): with
# oriented the object yaw the footprint rectangle was checked at, else 0
# (checked for any yaw).
geometry_msgs/PoseStamped pose
bool         oriented           # see pose; true for a request with approach_from

float32      clearance          # m from the center to the nearest obstacle, edge or unseen area
uint32       plane_id           # Plane.id of the support (in the response's planes)
uint32       surface_id         # Surface.id of the surface (/aha/perception/surfaces), 0 = not tracked yet
string       surface_label      # "table" | "shelf" | "floor" | "other"
float32      surface_height     # m, z of the position
float32      score              # clearance, or minus the distance to the request's near point. The list is
                                # ordered by the distance to near, else by clearance rounded to the grid
                                # resolution across planes (then the plane nearer the robot), so score
                                # is not strictly descending

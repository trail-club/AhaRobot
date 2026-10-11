# A free spot on a support plane where an object of the requested size fits.
# Owner: perception squad. Consumers: manipulation.

# Center of the object footprint on the surface. Orientation: z along the
# surface normal, yaw 0 (the footprint check is orientation free).
geometry_msgs/PoseStamped pose

float32      clearance          # m from the center to the nearest obstacle, edge or unseen area
uint32       plane_id           # Plane.id of the surface
uint32       surface_id         # Surface.id of the surface (/aha/perception/surfaces), 0 = not tracked yet
string       surface_label      # "table" | "shelf" | "floor" | "other"
float32      surface_height     # m, z of the position
float32      score              # higher first: clearance, or minus the distance to the request's near point

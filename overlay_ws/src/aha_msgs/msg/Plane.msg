# A plane detected in the head depth image (floor, table, shelf board, wall).
# Owner: perception squad. Consumers: manipulation.

std_msgs/Header header          # frame_id = output frame (base_footprint, z up)
uint32       id                 # index in the PlaneArray of the same detection
string       label              # "floor" | "table" | "shelf" | "wall" | "other"

# normal . p + offset = 0 in header.frame_id; horizontal planes point up.
geometry_msgs/Vector3 normal
float64      offset

geometry_msgs/Point   centroid
float64      area               # m^2, convex hull on the plane
geometry_msgs/Polygon hull      # convex hull vertices (3D) in header.frame_id
float64      height             # centroid z in header.frame_id (m)

# A surface (floor, table, shelf board, wall) tracked over the plane detections.
# Owner: perception squad. Consumers: task / manipulation.

uint32       id                 # stable over detections, never reused; 0 is not used
string       label              # majority of the recent labels: "floor" | "table" | "shelf" | "wall" | "other"
int32        level              # stacked horizontal surfaces sharing a footprint: 1 = lowest; 0 = not stacked
Plane        plane              # smoothed plane in the tracking frame (odom); plane.id = id
string[]     object_categories  # categories of objects seen on the surface (empty for now)
builtin_interfaces/Time last_seen  # depth image stamp of the last detection that observed it
uint32       observations       # detections that observed it
bool         visible            # observed in the latest detection

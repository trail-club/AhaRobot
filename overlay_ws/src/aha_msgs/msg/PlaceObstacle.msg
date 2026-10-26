# An object standing on a horizontal support plane, as a box for collision checks.
# Owner: perception squad. Consumers: manipulation.

# Box center in the frame of the response's planes (base_footprint); z up,
# x along the length of the minimum-area footprint rectangle.
geometry_msgs/Pose    pose
geometry_msgs/Vector3 size      # x length, y width (footprint), z height above the surface (m)
# Plane.id of the support in the same detection: any horizontal support of
# it, so possibly not in the response's planes (/aha/perception/planes of
# the response stamp has them all). Ignored in exclude_regions.
uint32                plane_id

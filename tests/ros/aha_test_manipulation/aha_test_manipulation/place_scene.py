"""move_group's planning scene for the place server.

Puts the perceived planes and obstacle boxes (aha_test_manipulation.scene_objects)
and the held object into the scene through /apply_planning_scene, and keeps
the robot's states valid against the obstacle boxes without deleting any
(/check_state_validity): a touching box loses its padding, then it is cut
clear of the touching link (top lowered or a side moved back), and a box
that would vanish is allowed against the touching links in the allowed
collision matrix.
"""

from __future__ import annotations

import numpy as np
from moveit_msgs.msg import PlanningScene, PlanningSceneComponents
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetStateValidity

from aha_test_manipulation import scene_objects as scene
from aha_test_manipulation import self_filter
from aha_test_manipulation.place_selection import PlaceError

ARMS = {
    side: {
        "group": f"{side}_arm",
        "joints": [f"joint_{side[0]}{i}" for i in range(1, 7)],
        "tcp": f"link_{side[0]}ee",
        "shoulder": f"link_{side[0]}2",
        "touch_links": [
            f"link_{side[0]}6",
            f"link_{side[0]}7r",
            f"link_{side[0]}7l",
            f"link_{side[0]}ee",
        ],
        "gripper_topic": f"/{side}_gripper_controller/commands",
        # Real arm: arm_node's gripper command (one finger, the other mirrors it).
        "gripper_joint_topic": f"/{side}/arm/gripper_joint_command",
        "gripper_joint": f"joint_{side[0]}7r",
        "gripper_state_topic": f"/{side}/arm/joint_states",  # arm_node's own
    }
    for side in ("right", "left")
}
PLACED_OBJECT_ID = "aha_placed_object"
MAX_CONTACT_PASSES = 10


def held_object_id(arm):
    return f"aha_held_object_{arm}"


def is_obstacle(object_id):
    return object_id.startswith(scene.OBSTACLE_PREFIX)


class PlaceScene:
    """Planning scene state of one place goal.

    ``node`` provides ``call(client, request, what)``, ``param(name)``,
    ``lookup(target, source)``, ``try_lookup(target, source)`` and
    ``get_logger()``; ``req`` arguments are the server's parsed goals (arm,
    grasp, size).
    """

    def __init__(self, node, callback_group):
        self.node = node
        self.apply_client = node.create_client(
            ApplyPlanningScene, "/apply_planning_scene", callback_group=callback_group
        )
        self.get_client = node.create_client(
            GetPlanningScene, "/get_planning_scene", callback_group=callback_group
        )
        self.validity_client = node.create_client(
            GetStateValidity, "/check_state_validity", callback_group=callback_group
        )
        self.boxes = {}  # obstacle id -> unpadded (maybe clipped) Box in the scene
        self.perceived = {}  # obstacle id -> Box as perceived (no padding)
        self.padded = set()  # obstacle ids still padded in the scene

    def param(self, name):
        return self.node.param(name)

    @property
    def frame(self):
        return self.param("planning_frame")

    # --- move_group services ----------------------------------------------------------

    def apply(self, world_objects=(), attached_objects=(), acm=None):
        diff = PlanningScene()
        diff.is_diff = True
        diff.robot_state.is_diff = True
        diff.world.collision_objects = list(world_objects)
        diff.robot_state.attached_collision_objects = list(attached_objects)
        if acm is not None:
            diff.allowed_collision_matrix = acm  # replaces the whole matrix
        request = ApplyPlanningScene.Request()
        request.scene = diff
        if not self.node.call(
            self.apply_client, request, "apply_planning_scene"
        ).success:
            raise PlaceError("apply_planning_scene failed")

    def get(self, components):
        request = GetPlanningScene.Request()
        request.components.components = components
        return self.node.call(self.get_client, request, "get_planning_scene").scene

    def object_ids(self):
        """(world object ids, attached object ids) in move_group's planning scene."""
        response = self.get(
            PlanningSceneComponents.WORLD_OBJECT_NAMES
            | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS
        )
        world = [obj.id for obj in response.world.collision_objects]
        attached = [
            obj.object.id for obj in response.robot_state.attached_collision_objects
        ]
        return world, attached

    def acm(self):
        return self.get(
            PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        ).allowed_collision_matrix

    def forget_allowed(self, forget):
        """Drop allowed collision matrix entries of ids that go or get reused."""
        acm = self.acm()
        if any(forget(name) for name in acm.entry_names):
            self.apply(acm=scene.acm_forget(acm, forget))

    def allow(self, object_id, bodies):
        self.apply(acm=scene.acm_allow(self.acm(), object_id, sorted(bodies)))

    def contacts(self, arm, is_object, joints=None):
        """{world object id: (robot bodies, contact points)} for the objects
        (``is_object(id)``) in collision with the current state (with
        ``joints`` {name: value} changed, if given)."""
        request = GetStateValidity.Request()
        request.group_name = ARMS[arm]["group"]
        request.robot_state.is_diff = True
        if joints:
            request.robot_state.joint_state.name = list(joints)
            request.robot_state.joint_state.position = [
                float(v) for v in joints.values()
            ]
        response = self.node.call(self.validity_client, request, "check_state_validity")
        out = {}
        if response.valid:
            return out
        for contact in response.contacts:
            pairs = (
                (contact.contact_body_1, contact.contact_body_2),
                (contact.contact_body_2, contact.contact_body_1),
            )
            for body, other in pairs:
                if is_object(body) and not is_object(other):
                    bodies, points = out.setdefault(body, (set(), []))
                    bodies.add(other)
                    p = contact.position
                    points.append((p.x, p.y, p.z))
        return out

    # --- perceived objects ------------------------------------------------------------

    def obstacle_objects(self, obstacles, keep=False):
        """Padded collision objects of PlaceObstacle messages, numbered after
        the kept ones (``keep``) or replacing the box state."""
        padding = self.param("obstacle_padding")
        if not keep:
            self.perceived, self.boxes, self.padded = {}, {}, set()
        objects = scene.obstacle_objects(
            [scene.obstacle_box(o, padding) for o in obstacles],
            self.frame,
            start=len(self.perceived),
        )
        for obj, obstacle in zip(objects, obstacles):
            self.perceived[obj.id] = self.boxes[obj.id] = scene.obstacle_box(obstacle)
            self.padded.add(obj.id)
        return objects

    def replace(self, planes, obstacles, req, scene_params):
        """Replace the perceived objects and (re)attach the held object;
        returns the boxes cut or allowed for the current state
        (``resolve_contacts``).

        The placed object of the previous goal is removed: it was kept in
        base_footprint, which moves with the robot, and perception reports
        it as an obstacle when it is seen. Allowed collision entries of
        these ids go too.

        A diff applies the attached objects before the world objects, so the
        world is cleaned in a separate diff: attaching an id that is still in
        the world would take that object instead.
        """
        frame = self.frame
        cfg = ARMS[req.arm]
        held_id = held_object_id(req.arm)
        world, attached = self.object_ids()

        def ours(object_id):
            return object_id.startswith(scene.PERCEIVED_PREFIXES) or object_id in (
                PLACED_OBJECT_ID,
                held_id,
            )

        stale = [scene.remove_object(i, frame) for i in world if ours(i)]
        if held_id in attached:
            # Detaching puts the object back into the world; remove it in the same diff.
            stale.append(scene.remove_object(held_id, frame))
            self.apply(stale, [scene.detach_object(held_id, cfg["tcp"])])
        elif stale:
            self.apply(stale)
        self.forget_allowed(ours)
        objects = scene.plane_objects(planes, frame, scene_params)
        boxes = self.obstacle_objects(obstacles)
        objects += boxes
        held = scene.held_object(
            held_id,
            cfg["tcp"],
            cfg["touch_links"],
            scene.BOX,
            req.size,
            req.hand,
        )
        self.apply(objects, [held])
        self.node.get_logger().info(
            f"planning scene: {len(objects)} perceived objects "
            f"({len(boxes)} obstacles), held object on {cfg['tcp']}"
        )
        return self.resolve_contacts(req)

    def add_obstacles(self, obstacles, req):
        """Add the obstacle boxes of a fresh detection (at the pre-place pose)
        to the ones in the scene; the older boxes stay, as the arm or a moved
        head may hide their objects now."""
        objects = self.obstacle_objects(obstacles, keep=True)
        if objects:
            self.apply(objects)
        self.resolve_contacts(req)

    # --- touching boxes ---------------------------------------------------------------

    def robot_points(self, req, bodies, contact_points):
        """Points (base frame) of the touching robot bodies: the box corners
        of arm links and of the held object, and the contact points."""
        points = [np.asarray(contact_points, dtype=float).reshape(-1, 3)]
        for body in bodies:
            box = None
            if body == held_object_id(req.arm):
                tcp = self.node.lookup(self.frame, ARMS[req.arm]["tcp"])
                box = self_filter.held_object_box(tcp, req.hand, req.size)
            elif self_filter.link_bounds(body) is not None:
                T, _ = self.node.try_lookup(self.frame, body)
                if T is not None:
                    box = self_filter.bounds_box(T, self_filter.link_bounds(body))
            if box is not None:
                points.append(scene.box_corners(box))
        return np.concatenate(points)

    def resolve_contacts(self, req):
        """Make the current state valid against the obstacle boxes, never
        removing one: unpad the touching boxes, then cut them
        ``clip_margin`` clear of the touching robot bodies (lower the top or
        move a side back, whichever keeps more of the box,
        ``scene_objects.clear_box``); a box that would keep less than
        ``min_box_height`` is allowed against the touching bodies and the
        touch links instead. Returns the ids of the boxes cut or allowed (the
        robot is at a real object there)."""
        touch = set(ARMS[req.arm]["touch_links"])
        unpadded, clipped, allowed = [], [], []
        for _ in range(MAX_CONTACT_PASSES):
            touched = self.contacts(req.arm, is_obstacle)
            if not touched:
                break
            diff = []
            for object_id, (bodies, contact_points) in sorted(touched.items()):
                box = self.boxes.get(object_id)
                if box is None:  # not from this goal's detections: never delete
                    allowed.append(object_id)
                    self.allow(object_id, bodies | touch)
                    continue
                if object_id in self.padded:
                    self.padded.discard(object_id)
                    unpadded.append(object_id)
                else:
                    cut = scene.clear_box(
                        box,
                        self.robot_points(req, bodies, contact_points),
                        self.param("clip_margin"),
                        self.param("min_box_height"),
                    )
                    if cut is None or cut is box:
                        allowed.append(object_id)
                        self.allow(object_id, bodies | touch)
                        continue
                    self.boxes[object_id] = box = cut
                    clipped.append(object_id)
                diff.append(
                    scene.collision_object(
                        object_id, self.frame, scene.BOX, box.size, box.pose
                    )
                )
            if diff:
                self.apply(diff)
        if unpadded or clipped:
            self.node.get_logger().info(
                f"obstacle boxes touching the robot: unpadded {sorted(set(unpadded))}, "
                f"cut {sorted(set(clipped))}"
            )
        if allowed:
            self.node.get_logger().warn(
                f"obstacle boxes {sorted(set(allowed))} touch the robot and cannot be "
                "cut: collisions with the touching links allowed"
            )
        return sorted(set(clipped) | set(allowed))

    def restore(self, object_ids, req):
        """Put back the perceived (unpadded) boxes of ``object_ids`` and drop
        their allowed collisions, then make the current state valid again."""
        ids = [i for i in object_ids if i in self.perceived]
        if not ids:
            return []
        self.forget_allowed(lambda name: name in ids)
        diff = []
        for object_id in ids:
            box = self.boxes[object_id] = self.perceived[object_id]
            diff.append(
                scene.collision_object(
                    object_id, self.frame, scene.BOX, box.size, box.pose
                )
            )
        self.apply(diff)
        return self.resolve_contacts(req)

    def allow_at(self, arm, joints, name):
        """Allow the perceived boxes and the placed object that collide with
        the arm at ``joints`` (a named state ``name``): seen under the arm,
        they would make the goal state invalid."""
        touched = self.contacts(
            arm, lambda n: is_obstacle(n) or n == PLACED_OBJECT_ID, joints
        )
        for object_id, (bodies, _) in sorted(touched.items()):
            self.allow(object_id, bodies)
        if touched:
            self.node.get_logger().warn(
                f"{sorted(touched)} touch the arm in {name}: collisions allowed"
            )

    # --- held and placed object -------------------------------------------------------

    def detach_held(self, arm):
        """Remove the held object from the scene (it is released)."""
        held_id = held_object_id(arm)
        _, attached = self.object_ids()
        if held_id in attached:
            self.apply(
                [scene.remove_object(held_id, self.frame)],
                [scene.detach_object(held_id, ARMS[arm]["tcp"])],
            )

    def add_placed(self, req, motion, padding=0.0):
        """Keep the released object as a world object at the place pose,
        ``padding`` larger on every side; where the hand still touches it,
        allow those collisions."""
        self.apply(
            [
                scene.collision_object(
                    PLACED_OBJECT_ID,
                    self.frame,
                    scene.BOX,
                    [float(v) + 2.0 * padding for v in req.size],
                    motion.object_pose,
                )
            ]
        )
        touched = self.contacts(req.arm, lambda name: name == PLACED_OBJECT_ID)
        if PLACED_OBJECT_ID in touched:
            bodies = touched[PLACED_OBJECT_ID][0] | set(ARMS[req.arm]["touch_links"])
            self.allow(PLACED_OBJECT_ID, bodies)
            self.node.get_logger().warn(
                f"{PLACED_OBJECT_ID} touches {sorted(bodies)}: collisions allowed"
            )

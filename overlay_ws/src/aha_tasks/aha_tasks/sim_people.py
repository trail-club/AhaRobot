"""Simulated people in fixed postures and gestures, without ROS or Gazebo.

The Fuel model "Male visitor" (OpenRobotics, version 2, CC0) is a rigged and
textured person in a T-pose. Its skin weights pose it here (linear blend
skinning) into static meshes for each label of person_perception: standing,
sitting, lying, waving (two frames shown in turn), raising an arm and pointing
sideways. The T-shirt colour is changed in the texture. The model is
downloaded once and checked by hash; generated meshes go to a cache directory
and Gazebo loads them by path.

Model frame: the person faces +x, their left is +y, z is up and the soles are
at z = 0 (a lying person's back is at z = 0, head towards -x).
"""

import colorsys
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

FUEL_URL = (
    "https://fuel.gazebosim.org/1.0/OpenRobotics/models/Male%20visitor/2/"
    "Male%20visitor.zip"
)
SOURCE_FILES = {
    "meshes/MaleVisitorWalk.dae": (
        "b86695b3520655d2954b377f5ddac2dd88d26ac673b6dd9cced0fffeeb4f97d3"
    ),
    "materials/textures/MaleVisitor.png": (
        "790ff35964cec374fee6a73f04db8ab8925504c6ced065c0dd7e04c6acc0e01d"
    ),
}
DAE, TEXTURE = SOURCE_FILES
COLLADA = "http://www.collada.org/2005/11/COLLADASchema"
NS = {"c": COLLADA}
ET.register_namespace("", COLLADA)
HIDDEN_Z = -5.0  # m; under the floor, like the HIC guests
GENERATOR_VERSION = 2  # raise when the generated files change for the same poses

X, Y, Z = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
ARMS_DOWN = [("UpperArm_L", X, -78.0), ("UpperArm_R", X, 78.0)]
LEFT_DOWN, RIGHT_DOWN = ARMS_DOWN
# (joint, axis in the current model frame, degrees) applied in order; a joint
# turns with everything below it, about its current position.
WAVE_UPPER_ARM = ("UpperArm_R", X, -35.0)  # out to the right and up
WAVE_FOREARM = -55.0  # then the forearm up
WAVE_SWING = 25.0  # degrees to each side
POSES = {
    "standing": ARMS_DOWN,
    "sitting": ARMS_DOWN
    + [
        ("UpperLeg_L", Y, -90.0),
        ("UpperLeg_R", Y, -90.0),
        ("LowerLeg_L", Y, 90.0),
        ("LowerLeg_R", Y, 90.0),
        ("LowerArm_L", Y, -70.0),  # hands on the thighs
        ("LowerArm_R", Y, -70.0),
    ],
    "raising_left_arm": [("UpperArm_L", X, 75.0), RIGHT_DOWN],
    "raising_right_arm": [LEFT_DOWN, ("UpperArm_R", X, -75.0)],
    "raising_both_arms": [("UpperArm_L", X, 75.0), ("UpperArm_R", X, -75.0)],
    "pointing_left": [("UpperArm_L", Z, -15.0), RIGHT_DOWN],
    "pointing_right": [LEFT_DOWN, ("UpperArm_R", Z, 15.0)],
    "waving_a": [
        LEFT_DOWN,
        WAVE_UPPER_ARM,
        ("LowerArm_R", X, WAVE_FOREARM + WAVE_SWING),
    ],
    "waving_b": [
        LEFT_DOWN,
        WAVE_UPPER_ARM,
        ("LowerArm_R", X, WAVE_FOREARM - WAVE_SWING),
    ],
}
# Whole-body rotation after the joints: on the back, head towards -x.
ROOT_ROTATIONS = {"lying": (Y, -90.0)}
LABELS = (
    "standing",
    "sitting",
    "lying",
    "waving",
    "raising_left_arm",
    "raising_right_arm",
    "raising_both_arms",
    "pointing_left",
    "pointing_right",
)
WAVE_FRAMES = ("waving_a", "waving_b")

# The T-shirt is the only purple in the texture.
SHIRT_HUE = (0.66, 0.82)
SHIRT_MIN_SATURATION = 0.2
SHIRT_COLOURS = {
    "purple": None,  # as modelled
    "red": (190, 25, 25),
    "blue": (30, 60, 190),
    "yellow": (230, 200, 30),
    "orange": (235, 120, 20),
    "green": (40, 140, 50),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "black": (30, 30, 30),
}
SCENE_KEYS = {"world", "wave_period", "people"}
PERSON_KEYS = {"label", "shirt", "position", "yaw"}


def mesh_frames(label):
    """Meshes of a label in display order; waving alternates two."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}; known: {', '.join(LABELS)}")
    return WAVE_FRAMES if label == "waving" else (label,)


# --- Source model ---------------------------------------------------------------


def fetch_source(cache: Path, url: str = FUEL_URL) -> Path:
    """The extracted Fuel model under `cache`, downloaded and checked once."""
    root = cache / "male_visitor_v2"
    if all(_sha256(root / name) == digest for name, digest in SOURCE_FILES.items()):
        return root
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    for name, digest in SOURCE_FILES.items():
        data = archive.read(name)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"{url}: {name} does not match the expected hash")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _sha256(path: Path):
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Skinning ---------------------------------------------------------------------


def _floats(element):
    return np.array(element.text.split(), dtype=float)


def _source(parent, source_id):
    return parent.find(f"c:source[@id='{source_id.lstrip('#')}']", NS)


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    angle = math.radians(degrees)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


class Rig:
    """The skinned mesh of the source model and its joint hierarchy."""

    def __init__(self, dae_path: Path):
        self.tree = ET.parse(dae_path)
        root = self.tree.getroot()
        mesh = root.find(".//c:library_geometries/c:geometry/c:mesh", NS)
        polylist = mesh.find("c:polylist", NS)
        inputs = {i.get("semantic"): i for i in polylist.findall("c:input", NS)}
        vertices = mesh.find("c:vertices", NS)
        position_id = vertices.find("c:input[@semantic='POSITION']", NS).get("source")
        self.positions_array = _source(mesh, position_id).find("c:float_array", NS)
        self.normals_array = _source(mesh, inputs["NORMAL"].get("source")).find(
            "c:float_array", NS
        )
        self.positions = _floats(self.positions_array).reshape(-1, 3)
        self.normals = _floats(self.normals_array).reshape(-1, 3)
        # The vertex of each normal, from the polygons' index tuples.
        stride = 1 + max(int(i.get("offset")) for i in inputs.values())
        corners = _floats(polylist.find("c:p", NS)).astype(int).reshape(-1, stride)
        vertex = corners[:, int(inputs["VERTEX"].get("offset"))]
        normal = corners[:, int(inputs["NORMAL"].get("offset"))]
        self.normal_vertex = np.zeros(len(self.normals), dtype=int)
        self.normal_vertex[normal] = vertex

        skin = root.find(".//c:library_controllers/c:controller/c:skin", NS)
        bind_shape = _floats(skin.find("c:bind_shape_matrix", NS)).reshape(4, 4)
        joints = skin.find("c:joints", NS)
        sources = {
            i.get("semantic"): _source(skin, i.get("source"))
            for i in joints.findall("c:input", NS)
        }
        self.joints = sources["JOINT"].find("c:Name_array", NS).text.split()
        inverse_bind = _floats(sources["INV_BIND_MATRIX"].find("c:float_array", NS))
        inverse_bind = inverse_bind.reshape(-1, 4, 4)
        # Rest pose = bind pose: joint origins in the model frame.
        self.pivots = np.array([np.linalg.inv(m)[:3, 3] for m in inverse_bind])
        if not np.allclose(bind_shape, np.eye(4)):
            raise ValueError("expected an identity bind shape matrix")

        weights_element = skin.find("c:vertex_weights", NS)
        w_inputs = {
            i.get("semantic"): i for i in weights_element.findall("c:input", NS)
        }
        weight_values = _floats(
            _source(skin, w_inputs["WEIGHT"].get("source")).find("c:float_array", NS)
        )
        counts = _floats(weights_element.find("c:vcount", NS)).astype(int)
        pairs = _floats(weights_element.find("c:v", NS)).astype(int).reshape(-1, 2)
        self.weights = np.zeros((len(self.positions), len(self.joints)))
        start = 0
        for v, count in enumerate(counts):
            for joint, weight in pairs[start : start + count]:
                self.weights[v, joint] += weight_values[weight]
            start += count
        self.weights /= np.maximum(self.weights.sum(axis=1, keepdims=True), 1e-9)

        self.parent = {}
        scene = root.find(".//c:library_visual_scenes/c:visual_scene", NS)

        def walk(node, parent):
            for child in node.findall("c:node", NS):
                sid = child.get("sid")
                if child.get("type") == "JOINT" and sid in self.joints:
                    self.parent[sid] = parent
                    walk(child, sid)
                else:
                    walk(child, parent)

        walk(scene, None)

    def subtree(self, joint):
        """The joint and every joint below it."""
        found = {joint}
        changed = True
        while changed:
            changed = False
            for child, parent in self.parent.items():
                if parent in found and child not in found:
                    found.add(child)
                    changed = True
        return found

    def pose(self, rotations, root_rotation=None):
        """Posed vertex positions and normals, soles (or back) at z = 0."""
        transforms = {joint: np.eye(4) for joint in self.joints}
        index = {joint: i for i, joint in enumerate(self.joints)}
        for joint, axis, degrees in rotations:
            pivot = transforms[joint] @ np.append(self.pivots[index[joint]], 1.0)
            turn = np.eye(4)
            turn[:3, :3] = rotation(axis, degrees)
            turn[:3, 3] = pivot[:3] - turn[:3, :3] @ pivot[:3]
            for moved in self.subtree(joint):
                transforms[moved] = turn @ transforms[moved]
        if root_rotation is not None:
            turn = np.eye(4)
            turn[:3, :3] = rotation(*root_rotation)
            transforms = {j: turn @ m for j, m in transforms.items()}
        stack = np.stack([transforms[joint] for joint in self.joints])  # (J, 4, 4)
        blended = np.einsum("vj,jab->vab", self.weights, stack)  # (V, 4, 4)
        homogeneous = np.column_stack([self.positions, np.ones(len(self.positions))])
        positions = np.einsum("vab,vb->va", blended, homogeneous)[:, :3]
        normals = np.einsum(
            "nab,nb->na", blended[self.normal_vertex, :3, :3], self.normals
        )
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        positions[:, 2] -= positions[:, 2].min()
        return positions, normals

    def static_dae(self, positions, normals, texture: str) -> bytes:
        """The model as a static mesh with these vertices, using `texture`."""
        root = ET.fromstring(ET.tostring(self.tree.getroot()))
        mesh = root.find(".//c:library_geometries/c:geometry/c:mesh", NS)
        geometry_id = root.find(".//c:library_geometries/c:geometry", NS).get("id")
        for array, values in (
            (self.positions_array, positions),
            (self.normals_array, normals),
        ):
            target = mesh.find(f".//c:float_array[@id='{array.get('id')}']", NS)
            target.text = " ".join(f"{v:.6g}" for v in values.ravel())
        for library in ("library_controllers", "library_animations"):
            element = root.find(f"c:{library}", NS)
            if element is not None:
                root.remove(element)
        root.find(".//c:library_images/c:image/c:init_from", NS).text = texture
        scene = root.find(".//c:library_visual_scenes/c:visual_scene", NS)
        # The skinned node may sit inside the armature; keep only it, at the top.
        # The armature's own transform is the identity in the source model.
        meshes = [
            node
            for node in scene.iter(f"{{{COLLADA}}}node")
            if node.find("c:instance_controller", NS) is not None
        ]
        for node in list(scene.findall("c:node", NS)):
            scene.remove(node)
        for node in meshes:
            controller = node.find("c:instance_controller", NS)
            node.remove(controller)
            for child in list(node.findall("c:node", NS)):
                node.remove(child)
            instance = ET.SubElement(node, f"{{{COLLADA}}}instance_geometry")
            instance.set("url", f"#{geometry_id}")
            for child in list(controller):
                if child.tag == f"{{{COLLADA}}}bind_material":
                    instance.append(child)
            scene.append(node)
        ET.register_namespace("", COLLADA)
        return ET.tostring(root, xml_declaration=True, encoding="utf-8")


# --- Texture -------------------------------------------------------------------------


def recolour_shirt(png: bytes, colour) -> bytes:
    """The texture with the purple T-shirt in `colour` (RGB), keeping its shading."""
    from PIL import Image

    image = Image.open(io.BytesIO(png)).convert("RGB")
    if colour is None:
        out = io.BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()
    rgb = np.asarray(image, dtype=np.float64) / 255.0
    hsv = np.array(
        [colorsys.rgb_to_hsv(*pixel) for pixel in rgb.reshape(-1, 3)]
    ).reshape(rgb.shape)
    shirt = (
        (hsv[..., 0] >= SHIRT_HUE[0])
        & (hsv[..., 0] <= SHIRT_HUE[1])
        & (hsv[..., 1] >= SHIRT_MIN_SATURATION)
    )
    value = hsv[..., 2]
    shade = value / max(float(np.median(value[shirt])), 1e-6)
    target = np.asarray(colour, dtype=np.float64) / 255.0
    recoloured = np.clip(shade[..., None] * target, 0.0, 1.0)
    rgb[shirt] = recoloured[shirt]
    out = io.BytesIO()
    Image.fromarray((rgb * 255).round().astype(np.uint8)).save(out, format="PNG")
    return out.getvalue()


# --- Models and scenes -----------------------------------------------------------------


def generation_key():
    """Changes with the poses, colours or output format, so stale meshes are not reused."""
    settings = json.dumps(
        [GENERATOR_VERSION, POSES, sorted(ROOT_ROTATIONS.items()), SHIRT_COLOURS],
        sort_keys=True,
    )
    return hashlib.sha256(settings.encode()).hexdigest()[:12]


def build_meshes(cache: Path, needed, url: str = FUEL_URL) -> dict:
    """{(mesh name, shirt): path} for the (label, shirt) pairs in `needed`."""
    source = fetch_source(cache, url)
    out = cache / f"posed_{generation_key()}"
    (out / "textures").mkdir(parents=True, exist_ok=True)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    rig = None
    paths = {}
    for label, shirt in sorted(set(needed)):
        if shirt not in SHIRT_COLOURS:
            raise ValueError(f"unknown shirt colour {shirt!r}")
        texture = out / "textures" / f"shirt_{shirt}.png"
        if not texture.is_file():
            texture.write_bytes(
                recolour_shirt((source / TEXTURE).read_bytes(), SHIRT_COLOURS[shirt])
            )
        for frame in mesh_frames(label):
            path = out / "meshes" / f"{frame}__{shirt}.dae"
            if not path.is_file():
                rig = rig or Rig(source / DAE)
                positions, normals = rig.pose(
                    POSES.get(frame, ARMS_DOWN), ROOT_ROTATIONS.get(frame)
                )
                path.write_bytes(
                    rig.static_dae(positions, normals, f"../textures/{texture.name}")
                )
            paths[(frame, shirt)] = path
    return paths


def person_sdf(name, mesh: Path, x, y, z, yaw):
    """A static visual without collision, so it cannot push the robot."""
    uri = "file://" + urllib.parse.quote(str(mesh))
    return (
        '<?xml version="1.0"?><sdf version="1.8">'
        f'<model name="{name}"><static>true</static>'
        f"<pose>{x} {y} {z} 0 0 {yaw}</pose>"
        f'<link name="link"><visual name="visual"><geometry><mesh>'
        f"<uri>{uri}</uri></mesh></geometry></visual></link></model></sdf>"
    )


def load_scene(path):
    """The people of a scene file; ValueError if it is incomplete."""
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(scene) != SCENE_KEYS:
        raise ValueError(f"{path}: needs exactly {sorted(SCENE_KEYS)}")
    if not scene["wave_period"] > 0:
        raise ValueError(f"{path}: wave_period must be positive")
    for name, person in scene["people"].items():
        if set(person) != PERSON_KEYS:
            raise ValueError(f"{path}: {name} needs exactly {sorted(PERSON_KEYS)}")
        mesh_frames(person["label"])
        if person["shirt"] not in SHIRT_COLOURS:
            raise ValueError(f"{path}: {name} has unknown shirt {person['shirt']!r}")
        if len(person["position"]) != 3:
            raise ValueError(f"{path}: {name} position must be [x, y, z]")
    return scene


def models(scene, paths):
    """(model name, person name, frame, mesh) for every model to spawn."""
    result = []
    for name, person in scene["people"].items():
        frames = mesh_frames(person["label"])
        for frame in frames:
            model = name if len(frames) == 1 else f"{name}__{frame}"
            result.append((model, name, frame, paths[(frame, person["shirt"])]))
    return result


def spawn_missing(missing, create, move):
    """Spawn each entry of `missing` (removed when done); returns those spawned.

    `create(entry)` adds the model; when it fails, `move(entry)` places a model
    of that name left from an earlier run in the same Gazebo. An entry that
    fails both stays missing without stopping the others.
    """
    spawned = []
    for entry in list(missing):
        if create(entry) or move(entry):
            missing.remove(entry)
            spawned.append(entry)
    return spawned


def shown_frame(label, sim_time, period):
    """The mesh frame of a label to show at a time; waving alternates."""
    frames = mesh_frames(label)
    return frames[int(sim_time / (period / len(frames))) % len(frames)]

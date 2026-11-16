"""Models of real objects for the simulation, built from public 3D datasets.

Two kinds, both built once into a cache at run time (nothing is committed):

- ScannedModel: a dataset's textured mesh (OBJ / PLY / DAE in an archive, or
  a GLB), downloaded and checked by its sha256, cropped, turned, scaled and
  centered on its bounding box (build_scanned).
- MeasuredModel: for objects without a scan, a measured shape (a revolved
  profile, a box, a metric scan or the profile measured from one) painted
  with images (build_measured).

The built directory has the layout of the tmc_wrs_gz_worlds YCB models
(model-1_4.sdf, meshes/<name>.obj, meshes/collision.stl): the collision is
the convex hull, the inertia that of the hull with uniform density.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import http.client
import io
import json
import math
from pathlib import Path
import posixpath
import struct
import tarfile
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

SCAN_BUILD_VERSION = 4  # raise when built models change
SCAN_SIZE_TOLERANCE = 0.002  # m, built bounding box against the catalog's size
SCAN_TEXTURE_SIZE = 2048  # px, largest side of a model's texture


class ModelError(ValueError):
    """A model that cannot be downloaded or built as described."""


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ------------------------------------------------------------ scanned models


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return _rotation(yaw) @ ry @ rx


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, sha256: str, cache: Path) -> Path:
    """The file at ``url`` in ``cache/downloads`` (fetched once), checked by its sha256."""
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    target = cache / "downloads" / f"{sha256[:12]}_{name}"
    if target.is_file() and _sha256(target) == sha256:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as reply, partial.open(
            "wb"
        ) as out:
            for block in iter(lambda: reply.read(1 << 20), b""):
                out.write(block)
    except (OSError, http.client.HTTPException) as error:  # e.g. IncompleteRead
        raise ModelError(f"cannot download {url}: {error}") from error
    actual = _sha256(partial)
    if actual != sha256:
        partial.unlink()
        raise ModelError(f"{url}: sha256 {actual}, expected {sha256}")
    partial.replace(target)
    return target


class _Archive:
    """Members of a zip or tar archive by path, or by file name if unique; a
    plain file is an archive of itself (member "")."""

    def __init__(self, path: Path):
        self._path = path
        if not zipfile.is_zipfile(path) and not tarfile.is_tarfile(path):
            self._zip, self._files = None, {}
        elif zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            self._files = {n: n for n in self._zip.namelist() if not n.endswith("/")}
        else:
            self._zip = None
            self._tar = tarfile.open(path)
            self._files = {m.name: m for m in self._tar.getmembers() if m.isfile()}

    def read(self, member: str, near: str = "") -> bytes:
        """The member; a relative ``member`` is looked up next to ``near`` first,
        then anywhere by its file name (scans keep textures in other folders)."""
        if not self._files and self._zip is None:
            if member:
                raise ModelError(
                    f"no {member!r} next to the single file {self._path.name}"
                )
            return self._path.read_bytes()
        wanted = [posixpath.normpath(posixpath.join(posixpath.dirname(near), member))]
        name = posixpath.basename(member)
        wanted += sorted(p for p in self._files if posixpath.basename(p) == name)
        for path in wanted:
            if path in self._files:
                if self._zip is not None:
                    return self._zip.read(path)
                return self._tar.extractfile(self._files[path]).read()
        raise ModelError(f"no {member!r} in the archive")


@dataclass
class Mesh:
    """A textured triangle mesh read from a scan.

    corners (n, 3, 3): the 1-based vertex / uv / normal index of each corner
    of each triangle (0: none); material (n,): index into materials, each a
    dict with ``texture`` (image bytes or None) and ``color`` (r, g, b in 0-1)."""

    vertices: np.ndarray
    uvs: np.ndarray
    normals: np.ndarray
    corners: np.ndarray
    material: np.ndarray
    materials: list


def _mesh(vertices, uvs, normals, corners, material, materials) -> Mesh:
    if not len(vertices) or not len(corners):
        raise ModelError("empty mesh")
    return Mesh(
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(uvs, dtype=float).reshape(-1, 2),
        np.asarray(normals, dtype=float).reshape(-1, 3),
        np.asarray(corners, dtype=int).reshape(-1, 3, 3),
        np.asarray(material, dtype=int).reshape(-1),
        materials,
    )


def _parse_mtl(text: str) -> dict:
    """name -> (diffuse texture file or None, Kd colour) of a Wavefront MTL."""
    out, name = {}, None
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        key, value = parts[0], parts[1].strip()
        if key == "newmtl":
            name = value
            out[name] = [None, (1.0, 1.0, 1.0)]
        elif name is not None and key == "map_Kd":
            out[name][0] = value.split()[-1]
        elif name is not None and key == "Kd":
            out[name][1] = tuple(float(v) for v in value.split()[:3])
    return out


def read_obj(archive, member: str) -> Mesh:
    """A Wavefront OBJ of the archive with its MTL materials and textures;
    polygons are fanned into triangles."""
    vertices, uvs, normals, corners, material = [], [], [], [], []
    names, current, library = [], 0, {}
    for line in archive.read(member).decode("utf-8", "replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "v":
            vertices.append([float(v) for v in parts[1:4]])
        elif key == "vt":
            uvs.append([float(v) for v in parts[1:3]])
        elif key == "vn":
            normals.append([float(v) for v in parts[1:4]])
        elif key == "mtllib":
            text = archive.read(line.split(None, 1)[1].strip(), near=member)
            library.update(_parse_mtl(text.decode("utf-8", "replace")))
        elif key == "usemtl":
            name = line.split(None, 1)[1].strip() if len(parts) > 1 else ""
            if name not in names:
                names.append(name)
            current = names.index(name)
        elif key == "f":
            face = []
            for corner in parts[1:]:
                fields = (corner.split("/") + ["", ""])[:3]
                index = []
                for value, count in zip(fields, (vertices, uvs, normals)):
                    k = int(value) if value else 0
                    index.append(k + len(count) + 1 if k < 0 else k)
                face.append(index)
            for k in range(1, len(face) - 1):
                corners.append([face[0], face[k], face[k + 1]])
                material.append(current)
    if not names:  # faces before any usemtl: the library's only material
        names = list(library)[:1] or [""]
    materials = []
    for name in names:
        texture, color = library.get(name, (None, (1.0, 1.0, 1.0)))
        image = archive.read(texture, near=member) if texture else None
        materials.append({"texture": image, "color": color})
    return _mesh(vertices, uvs, normals, corners, material, materials)


def read_ply(archive, member: str) -> Mesh:
    """An ASCII PLY with per-vertex texture_u / texture_v and a
    ``comment TextureFile`` (the BOP format of HOPE)."""
    lines = iter(archive.read(member).decode("utf-8", "replace").splitlines())
    if next(lines).strip() != "ply":
        raise ModelError(f"{member}: not a PLY")
    texture, counts, properties, element = None, {}, {}, None
    for line in lines:
        parts = line.split()
        if parts[:2] == ["format", "ascii"]:
            continue
        if parts[0] == "format":
            raise ModelError(f"{member}: only ASCII PLY is read")
        if parts[:2] == ["comment", "TextureFile"]:
            texture = parts[2]
        elif parts[0] == "element":
            element = parts[1]
            counts[element] = int(parts[2])
            properties[element] = []
        elif parts[0] == "property":
            properties[element].append(parts[-1])
        elif parts[0] == "end_header":
            break
    names = properties["vertex"]
    rows = np.array(
        [next(lines).split()[: len(names)] for _ in range(counts["vertex"])],
        dtype=float,
    )
    column = {n: rows[:, k] for k, n in enumerate(names)}
    vertices = np.stack([column["x"], column["y"], column["z"]], axis=1)
    uvs = np.stack([column["texture_u"], column["texture_v"]], axis=1)
    normals = (
        np.stack([column["nx"], column["ny"], column["nz"]], axis=1)
        if "nx" in column
        else np.zeros((0, 3))
    )
    faces = []
    for _ in range(counts["face"]):
        values = [int(v) for v in next(lines).split()]
        polygon = values[1 : 1 + values[0]]
        for k in range(1, len(polygon) - 1):
            faces.append([polygon[0], polygon[k], polygon[k + 1]])
    faces = np.array(faces, dtype=int) + 1
    has_normals = 1 if len(normals) else 0
    corners = np.stack([faces, faces, faces * has_normals], axis=2)
    image = archive.read(texture, near=member) if texture else None
    materials = [{"texture": image, "color": (1.0, 1.0, 1.0)}]
    return _mesh(vertices, uvs, normals, corners, np.zeros(len(faces)), materials)


def read_dae(archive, member: str) -> Mesh:
    """The triangles and polylists of a COLLADA file as stored (its unit and
    up axis are not applied), node transforms applied, with the diffuse
    texture or colour of each material."""
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.fromstring(archive.read(member))

    def by_id(tag, ref):
        found = root.find(f".//c:{tag}[@id='{ref.lstrip('#')}']", ns)
        if found is None:
            raise ModelError(f"{member}: no {tag} {ref}")
        return found

    def source(ref):
        element = by_id("source", ref)
        stride = int(element.find("c:technique_common/c:accessor", ns).get("stride", 1))
        values = np.array(
            element.findtext("c:float_array", "", ns).split(), dtype=float
        )
        return values.reshape(-1, stride)

    def material(symbol):
        element = by_id("material", symbol)
        effect = by_id("effect", element.find("c:instance_effect", ns).get("url"))
        diffuse = effect.find(".//c:diffuse", ns)
        texture = diffuse.find("c:texture", ns) if diffuse is not None else None
        if texture is None:
            color = (
                diffuse.findtext("c:color", "1 1 1 1", ns)
                if diffuse is not None
                else ""
            )
            return {
                "texture": None,
                "color": tuple(float(v) for v in (color.split() or [1, 1, 1])[:3]),
            }
        params = {p.get("sid"): p for p in effect.iter(f"{{{ns['c']}}}newparam")}
        name = texture.get("texture")
        if name in params:  # sampler2D -> surface -> image
            surface = params[params[name].findtext(".//c:source", "", ns)]
            name = surface.findtext(".//c:init_from", "", ns)
        path = by_id("image", name).findtext("c:init_from", "", ns).strip()
        return {
            "texture": archive.read(urllib.parse.unquote(path), near=member),
            "color": (1.0, 1.0, 1.0),
        }

    vertices, uvs, normals, corners, face_material, materials, names = (
        [],
        [],
        [],
        [],
        [],
        [],
        {},
    )
    counts = [0, 0, 0]

    def add_geometry(geometry, matrix, bindings):
        mesh = geometry.find("c:mesh", ns)
        for primitive in [
            *mesh.findall("c:triangles", ns),
            *mesh.findall("c:polylist", ns),
        ]:
            inputs = primitive.findall("c:input", ns)
            width = max(int(i.get("offset")) for i in inputs) + 1
            indices = np.array(primitive.findtext("c:p", "", ns).split(), dtype=int)
            indices = indices.reshape(-1, width)
            if primitive.tag.endswith("polylist"):
                sizes = np.array(
                    primitive.findtext("c:vcount", "", ns).split(), dtype=int
                )
                starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
                rows = [
                    [s, s + k, s + k + 1]
                    for s, n in zip(starts, sizes)
                    for k in range(1, n - 1)
                ]
            else:
                rows = np.arange(len(indices)).reshape(-1, 3)
            corner = np.zeros((len(rows), 3, 3), dtype=int)
            for item in inputs:
                semantic, offset = item.get("semantic"), int(item.get("offset"))
                if semantic == "VERTEX":
                    ref = by_id("vertices", item.get("source")).find(
                        "c:input[@semantic='POSITION']", ns
                    )
                    data = source(ref.get("source"))
                    vertices.append(data @ matrix[:3, :3].T + matrix[:3, 3])
                    corner[:, :, 0] = indices[rows, offset] + counts[0] + 1
                    counts[0] += len(data)
                elif semantic == "NORMAL":
                    data = source(item.get("source"))
                    normals.append(data @ np.linalg.inv(matrix[:3, :3]))
                    corner[:, :, 2] = indices[rows, offset] + counts[2] + 1
                    counts[2] += len(data)
                elif semantic == "TEXCOORD" and item.get("set", "0") in ("0", None):
                    data = source(item.get("source"))[:, :2]
                    uvs.append(data)
                    corner[:, :, 1] = indices[rows, offset] + counts[1] + 1
                    counts[1] += len(data)
            symbol = bindings.get(primitive.get("material"), primitive.get("material"))
            if symbol not in names:
                names[symbol] = len(materials)
                materials.append(
                    material(symbol)
                    if symbol
                    else {"texture": None, "color": (1.0, 1.0, 1.0)}
                )
            corners.append(corner)
            face_material.append(np.full(len(corner), names[symbol]))

    def node_matrix(node):
        matrix = np.eye(4)
        for item in node:
            tag = item.tag.split("}")[-1]
            values = [float(v) for v in (item.text or "").split()]
            step = np.eye(4)
            if tag == "matrix":
                step = np.array(values).reshape(4, 4)
            elif tag == "translate":
                step[:3, 3] = values
            elif tag == "scale":
                step[:3, :3] = np.diag(values)
            elif tag == "rotate":
                x, y, z = np.array(values[:3]) / np.linalg.norm(values[:3])
                c, s = (
                    math.cos(math.radians(values[3])),
                    math.sin(math.radians(values[3])),
                )
                k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                step[:3, :3] = np.eye(3) + s * k + (1 - c) * k @ k
            matrix = matrix @ step
        return matrix

    def visit(node, parent):
        matrix = parent @ node_matrix(node)
        for instance in node.findall("c:instance_geometry", ns):
            bindings = {
                b.get("symbol"): b.get("target").lstrip("#")
                for b in instance.iter(f"{{{ns['c']}}}instance_material")
            }
            add_geometry(by_id("geometry", instance.get("url")), matrix, bindings)
        for child in node.findall("c:node", ns):
            visit(child, matrix)

    scene = root.find("c:library_visual_scenes/c:visual_scene", ns)
    if scene is not None:
        for node in scene.findall("c:node", ns):
            visit(node, np.eye(4))
    else:
        for geometry in root.findall(".//c:geometry", ns):
            add_geometry(geometry, np.eye(4), {})
    if not corners:
        raise ModelError(f"{member}: no triangles")
    return _mesh(
        np.concatenate(vertices),
        np.concatenate(uvs) if uvs else np.zeros((0, 2)),
        np.concatenate(normals) if normals else np.zeros((0, 3)),
        np.concatenate(corners),
        np.concatenate(face_material),
        materials,
    )


GLTF_TYPES = {
    5120: "<i1",
    5121: "<u1",
    5122: "<i2",
    5123: "<u2",
    5125: "<u4",
    5126: "<f4",
}
GLTF_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _node_matrix(node) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0.0, 0.0, 0.0, 1.0))
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(node.get("scale", (1.0, 1.0, 1.0)))
    matrix[:3, 3] = node.get("translation", (0.0, 0.0, 0.0))
    return matrix


def read_glb(data: bytes) -> Mesh:
    """The triangles of a binary glTF (y up, as stored), node transforms
    applied, with the base colour texture or factor of each material."""
    magic, _, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF":
        raise ModelError("not a binary glTF")
    offset, gltf, binary = 12, None, b""
    while offset < len(data):
        length, kind = struct.unpack_from("<I4s", data, offset)
        chunk = data[offset + 8 : offset + 8 + length]
        if kind == b"JSON":
            gltf = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        offset += 8 + length
    views = gltf.get("bufferViews", [])

    def view(index):
        v = views[index]
        start = v.get("byteOffset", 0)
        return binary[start : start + v["byteLength"]], v.get("byteStride")

    def accessor(index):
        a = gltf["accessors"][index]
        if "sparse" in a or "bufferView" not in a:
            raise ModelError("sparse glTF accessors are not read")
        raw, stride = view(a["bufferView"])
        dtype = np.dtype(GLTF_TYPES[a["componentType"]])
        width = GLTF_SIZES[a["type"]]
        start = a.get("byteOffset", 0)
        step = stride or dtype.itemsize * width
        rows = np.ndarray(
            (a["count"], width),
            dtype=dtype,
            buffer=raw,
            offset=start,
            strides=(step, dtype.itemsize),
        ).astype(float)
        if a.get("normalized"):
            rows /= float(np.iinfo(dtype).max)
        return rows

    materials, material_index = [], {}

    def material(index):
        if index in material_index:
            return material_index[index]
        spec = gltf["materials"][index] if index is not None else {}
        pbr = spec.get("pbrMetallicRoughness", {})
        color = tuple(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0))[:3])
        image = None
        if "baseColorTexture" in pbr:
            if pbr["baseColorTexture"].get("texCoord", 0):
                raise ModelError("glTF base colour texture on a second uv set")
            source = gltf["textures"][pbr["baseColorTexture"]["index"]]["source"]
            image = view(gltf["images"][source]["bufferView"])[0]
        material_index[index] = len(materials)
        materials.append({"texture": image, "color": color})
        return material_index[index]

    vertices, uvs, normals, corners, face_material = [], [], [], [], []
    count = 0

    def visit(index, parent):
        nonlocal count
        node = gltf["nodes"][index]
        matrix = parent @ _node_matrix(node)
        for primitive in (
            gltf["meshes"][node["mesh"]]["primitives"] if "mesh" in node else []
        ):
            if primitive.get("mode", 4) != 4:
                continue
            if "KHR_draco_mesh_compression" in primitive.get("extensions", {}):
                raise ModelError("Draco compressed glTF is not read")
            attributes = primitive["attributes"]
            points = accessor(attributes["POSITION"])
            n = len(points)
            vertices.append(points @ matrix[:3, :3].T + matrix[:3, 3])
            if "TEXCOORD_0" in attributes:
                uv = accessor(attributes["TEXCOORD_0"])
                uvs.append(np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1))
            else:
                uvs.append(np.zeros((n, 2)))
            if "NORMAL" in attributes:
                normal = accessor(attributes["NORMAL"]) @ np.linalg.inv(matrix[:3, :3])
                normals.append(normal)
            else:
                normals.append(np.zeros((n, 3)))
            if "indices" in primitive:
                tri = accessor(primitive["indices"]).astype(int).reshape(-1, 3)
            else:
                tri = np.arange(n).reshape(-1, 3)
            if np.linalg.det(matrix[:3, :3]) < 0:
                tri = tri[:, [0, 2, 1]]
            tri = tri + count + 1
            corners.append(np.stack([tri, tri, tri], axis=2))
            face_material.append(np.full(len(tri), material(primitive.get("material"))))
            count += n
        for child in node.get("children", []):
            visit(child, matrix)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for root in scene.get("nodes", range(len(gltf["nodes"]))):
        visit(root, np.eye(4))
    if not corners:
        raise ModelError("glTF without triangles")
    return _mesh(
        np.concatenate(vertices),
        np.concatenate(uvs),
        np.concatenate(normals),
        np.concatenate(corners),
        np.concatenate(face_material),
        materials,
    )


def crop_mesh(mesh: Mesh, lower, upper) -> Mesh:
    """The triangles with every corner inside the box, unused vertices dropped."""
    inside = np.all((mesh.vertices >= lower) & (mesh.vertices <= upper), axis=1)
    keep = np.all(inside[mesh.corners[:, :, 0] - 1], axis=1)
    if not keep.any():
        raise ModelError("crop keeps no triangle")
    corners = mesh.corners[keep]
    used = np.unique(corners[:, :, 0])
    remap = np.zeros(len(mesh.vertices) + 1, dtype=int)
    remap[used] = np.arange(1, len(used) + 1)
    corners = corners.copy()
    corners[:, :, 0] = remap[corners[:, :, 0]]
    return Mesh(
        mesh.vertices[used - 1],
        mesh.uvs,
        mesh.normals,
        corners,
        mesh.material[keep],
        mesh.materials,
    )


def _orient_mesh(vertices, normals, corners):
    """(normals, corners) with the triangles wound outwards and valid normals.

    Some scans (the Berkeley meshes of YCB) are wound inwards, which Gazebo
    culls as back faces, and index normals they do not have: such normals are
    replaced by area-weighted vertex normals."""
    if corners[:, :, 0].min() < 1 or corners[:, :, 0].max() > len(vertices):
        raise ModelError("face refers to a missing vertex")
    a, b, c = (vertices[corners[:, k, 0] - 1] for k in range(3))
    center = vertices.mean(axis=0)
    volume = np.einsum("ij,ij->i", a - center, np.cross(b - center, c - center))
    if volume.sum() < 0:
        corners = corners[:, [0, 2, 1]]
        b, c = c, b
    faces = np.cross(b - a, c - a)
    indices = corners[:, :, 2]
    valid = len(normals) and indices.min() >= 1 and indices.max() <= len(normals)
    if valid:
        given = normals[indices - 1].sum(axis=1)
        valid = np.mean(np.einsum("ij,ij->i", given, faces) > 0) > 0.9
    if not valid:
        normals = np.zeros_like(vertices)
        for k in range(3):
            np.add.at(normals, corners[:, k, 0] - 1, faces)
        corners = corners.copy()
        corners[:, :, 2] = corners[:, :, 0]
    normals = normals / np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
    )
    return normals, corners


def _outward_hull(points: np.ndarray) -> np.ndarray:
    """(n, 3) triangles of the convex hull of ``points``, wound outwards."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    triangles = hull.simplices.copy()
    inside = points[hull.vertices].mean(axis=0)
    a, b, c = (points[triangles[:, k]] for k in range(3))
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), a - inside) < 0
    triangles[flip] = triangles[flip][:, [0, 2, 1]]
    return triangles


def solid_inertia(points: np.ndarray, triangles: np.ndarray, mass: float):
    """(center of mass, 3x3 inertia about it) of a closed outward mesh of uniform density."""
    a, b, c = (points[triangles[:, k]] for k in range(3))
    volumes = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    volume = volumes.sum()
    if volume <= 0:
        raise ModelError("mesh encloses no volume")
    center = (volumes[:, None] * (a + b + c)).sum(axis=0) / (4.0 * volume)
    s = a + b + c
    second = np.einsum("i,ij,ik->jk", volumes / 20.0, a, a)
    second += np.einsum("i,ij,ik->jk", volumes / 20.0, b, b)
    second += np.einsum("i,ij,ik->jk", volumes / 20.0, c, c)
    second += np.einsum("i,ij,ik->jk", volumes / 20.0, s, s)
    second = second * (mass / volume) - mass * np.outer(center, center)
    return center, np.trace(second) * np.eye(3) - second


def _write_stl(path: Path, points: np.ndarray, triangles: np.ndarray) -> None:
    a, b, c = (points[triangles[:, k]] for k in range(3))
    normals = np.cross(b - a, c - a)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rows = np.zeros(
        len(triangles), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]
    )
    rows["n"] = normals
    rows["v"] = np.stack([a, b, c], axis=1)
    path.write_bytes(
        b"aha_tasks.sim_objects".ljust(80, b" ")
        + struct.pack("<I", len(triangles))
        + rows.tobytes()
    )


def _write_scan(mesh: Mesh, name: str, directory: Path) -> None:
    """``<name>.obj`` / ``.mtl`` and one ``<name>_<k>.png`` per textured
    material; every material is named after the model (Gazebo keeps a mesh
    material by its name across models)."""
    from PIL import Image

    lines = [
        "# generated by aha_tasks.sim_objects from a scanned model",
        f"mtllib {name}.mtl",
    ]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in mesh.vertices]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in mesh.uvs]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.normals]
    library = []
    for k, material in enumerate(mesh.materials):
        faces = np.flatnonzero(mesh.material == k)
        if not len(faces):
            continue
        label = f"{name}_{k}"
        r, g, b = material["color"]
        entry = [f"newmtl {label}", "Ka 1 1 1", "Ks 0.05 0.05 0.05", "Ns 10", "illum 2"]
        if material["texture"] is not None:
            image = Image.open(io.BytesIO(material["texture"])).convert("RGB")
            if max(image.size) > SCAN_TEXTURE_SIZE:
                image.thumbnail((SCAN_TEXTURE_SIZE, SCAN_TEXTURE_SIZE), Image.LANCZOS)
            image.save(directory / f"{label}.png")
            entry += [f"Kd {r:.4f} {g:.4f} {b:.4f}", f"map_Kd {label}.png"]
        else:
            entry += [f"Kd {r:.4f} {g:.4f} {b:.4f}"]
        library.append("\n".join(entry))
        lines.append(f"usemtl {label}")
        for triangle in mesh.corners[faces]:
            refs = []
            for v, t, n in triangle:
                if t and n:
                    refs.append(f"{v}/{t}/{n}")
                elif t:
                    refs.append(f"{v}/{t}")
                elif n:
                    refs.append(f"{v}//{n}")
                else:
                    refs.append(f"{v}")
            lines.append("f " + " ".join(refs))
    (directory / f"{name}.mtl").write_text("\n\n".join(library) + "\n")
    (directory / f"{name}.obj").write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class ScannedModel:
    label: str
    model: str  # cache directory and model name
    url: str  # the model: an archive with an OBJ / PLY / DAE, or a GLB
    sha256: str
    mesh: str  # OBJ / PLY / DAE inside the archive; empty for a GLB
    rpy: tuple[float, float, float]  # rad, turn of the scan: z up, sides along x and y
    scale: float  # scan units to m
    crop: tuple | None  # (lower, upper) x, y, z of the scan's part to keep, or None
    mass: float  # kg
    size: tuple[float, float, float]  # m, bounding box of the built model
    source: str  # dataset, object and licence (credited in the built model)

    def entry(self) -> dict:
        return {
            "version": SCAN_BUILD_VERSION,
            "url": self.url,
            "sha256": self.sha256,
            "mesh": self.mesh,
            "rpy": list(self.rpy),
            "scale": self.scale,
            "crop": [list(c) for c in self.crop] if self.crop else None,
            "mass": self.mass,
            "size": list(self.size),
            "source": self.source,
        }


def read_scan(path: Path, member: str) -> Mesh:
    """The mesh of a downloaded scan: a GLB file, or an OBJ / PLY / DAE member of an archive."""
    with path.open("rb") as stream:
        magic = stream.read(4)
    if magic == b"glTF":
        return read_glb(path.read_bytes())
    archive = _Archive(path)
    if member.lower().endswith(".ply"):
        return read_ply(archive, member)
    if member.lower().endswith(".dae"):
        return read_dae(archive, member)
    return read_obj(archive, member)


def _finish(mesh: Mesh, size, label: str) -> np.ndarray:
    """Move the mesh's bounding box center to the origin, orient it and check
    its size against the catalog; the vertices in use."""
    used = mesh.vertices[np.unique(mesh.corners[:, :, 0]) - 1]
    lower, upper = used.min(axis=0), used.max(axis=0)
    mesh.vertices = mesh.vertices - (lower + upper) / 2
    mesh.normals, mesh.corners = _orient_mesh(mesh.vertices, mesh.normals, mesh.corners)
    if np.max(np.abs((upper - lower) - np.array(size))) > SCAN_SIZE_TOLERANCE:
        raise ModelError(
            f"{label}: built size {np.round(upper - lower, 4).tolist()} m,"
            f" catalog {list(size)}"
        )
    return mesh.vertices[np.unique(mesh.corners[:, :, 0]) - 1]


def _write_model(
    directory: Path, name: str, mesh: Mesh, mass: float, credit: str
) -> None:
    """The Gazebo model directory: visual mesh, convex hull collision and the
    inertia of the hull for ``mass``."""
    used = mesh.vertices[np.unique(mesh.corners[:, :, 0]) - 1]
    hull = _outward_hull(used)
    center, inertia = solid_inertia(used, hull, mass)
    meshes = directory / "meshes"
    meshes.mkdir(parents=True, exist_ok=True)
    for old in meshes.iterdir():
        old.unlink()
    _write_scan(mesh, name, meshes)
    _write_stl(meshes / "collision.stl", used, hull)
    i = inertia
    (directory / "model-1_4.sdf").write_text(
        "<?xml version='1.0'?>\n"
        f"<!-- {credit.splitlines()[0]}; built by aha_tasks.sim_objects -->\n"
        f"<sdf version='1.8'><model name='{name}'><link name='body'>"
        f"<inertial><mass>{mass}</mass>"
        f"<pose>{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} 0 0 0</pose>"
        f"<inertia><ixx>{i[0, 0]:.9g}</ixx><ixy>{i[0, 1]:.9g}</ixy>"
        f"<ixz>{i[0, 2]:.9g}</ixz><iyy>{i[1, 1]:.9g}</iyy><iyz>{i[1, 2]:.9g}</iyz>"
        f"<izz>{i[2, 2]:.9g}</izz></inertia></inertial>"
        f"<visual name='visual'><geometry><mesh><uri>model://{name}/meshes/{name}.obj"
        "</uri></mesh></geometry></visual>"
        f"<collision name='collision'><geometry><mesh><uri>model://{name}/meshes/"
        "collision.stl</uri></mesh></geometry></collision>"
        "</link></model></sdf>\n"
    )
    (directory / "SOURCE.txt").write_text(credit + "\n")


def _built(directory: Path, entry: dict) -> bool:
    stamp = directory / "build.json"
    return stamp.is_file() and json.loads(stamp.read_text()) == entry


def _stamp(directory: Path, entry: dict) -> None:
    (directory / "build.json").write_text(json.dumps(entry))


def build_scanned(model: ScannedModel, cache: Path) -> Path:
    """Build (once) a Gazebo model directory from the scan, in the layout of
    the tmc_wrs_gz_worlds YCB models; returns the directory.

    The scan is cropped to the box ``crop`` (scan coordinates, e.g. to cut
    off the turntable), turned by ``rpy``, scaled and moved so that its
    bounding box center is the origin. Visual: the scan's mesh and textures
    (at most ``SCAN_TEXTURE_SIZE`` pixels). Collision: its convex hull, from
    which the inertia follows for ``mass``. The bounding box must be ``size``."""
    directory = cache / model.model
    if _built(directory, model.entry()):
        return directory
    mesh = read_scan(download(model.url, model.sha256, cache), model.mesh)
    if model.crop is not None:
        mesh = crop_mesh(mesh, *(np.array(c, dtype=float) for c in model.crop))
    turn = _rpy_matrix(*model.rpy)
    mesh.vertices = mesh.vertices @ turn.T * model.scale
    if len(mesh.normals):
        mesh.normals = mesh.normals @ turn.T
    _finish(mesh, model.size, model.label)
    (directory / "build.json").unlink(missing_ok=True)
    _write_model(
        directory, model.model, mesh, model.mass, f"{model.source}\n{model.url}"
    )
    _stamp(directory, model.entry())
    return directory


# ----------------------------------------------------------- measured models

LABEL_WIDTH = 2048  # px of the texture atlas, a side label's 360 deg
SEGMENTS = 96  # around a revolved shape
FACE_AXES = {
    "+x": (0, 1),
    "-x": (0, -1),
    "+y": (1, 1),
    "-y": (1, -1),
    "+z": (2, 1),
    "-z": (2, -1),
}
# (right, up) of each box face seen from outside: z up on the sides, +y / -y
# up on the top / bottom; right x up is the outward normal.
FACE_FRAMES = {
    "+x": ((0, 1, 0), (0, 0, 1)),
    "-x": ((0, -1, 0), (0, 0, 1)),
    "+y": ((-1, 0, 0), (0, 0, 1)),
    "-y": ((1, 0, 0), (0, 0, 1)),
    "+z": ((1, 0, 0), (0, 1, 0)),
    "-z": ((1, 0, 0), (0, -1, 0)),
}


def _rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    if len(text) != 6:
        raise ModelError(f"colour must be #rrggbb: {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


LATIN_FONTS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
)
CJK_FONTS = (
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
)


def font_files() -> dict:
    """The font files that draw Latin and Japanese text here (None: missing)."""
    found = {}
    for kind, paths in (("latin", LATIN_FONTS), ("cjk", CJK_FONTS)):
        found[kind] = next((p for p in paths if Path(p).is_file()), None)
    return found


def _font(cjk: bool, size: int):
    from PIL import ImageFont

    path = font_files()["cjk" if cjk else "latin"]
    if path is None:  # a default font would neither scale nor draw Japanese
        kind = "Japanese" if cjk else "Latin"
        raise ModelError(
            f"no {kind} font: install one of {CJK_FONTS if cjk else LATIN_FONTS}"
        )
    return ImageFont.truetype(path, size)


def _runs(line: str):
    """(text, cjk) runs of a line: Japanese and Latin text use different fonts."""
    runs = []
    for c in line:
        cjk = ord(c) > 0x2FF
        if runs and runs[-1][1] == cjk:
            runs[-1][0] += c
        else:
            runs.append([c, cjk])
    return runs


def _text(draw, box, text: str, colour) -> None:
    """``text`` as large as fits into ``box`` (px), centered; "\n" breaks lines."""
    x0, y0, x1, y1 = box
    lines = text.split("\n")

    def width(line, size):
        return sum(draw.textlength(t, font=_font(cjk, size)) for t, cjk in _runs(line))

    low, high = 4, max(int(y1 - y0), 5)
    while low < high:  # the largest font size that fits
        size = (low + high + 1) // 2
        fits = max(width(line, size) for line in lines) <= x1 - x0
        if fits and size * 1.15 * len(lines) <= y1 - y0:
            low = size
        else:
            high = size - 1
    step = low * 1.15
    y = (y0 + y1) / 2 - step * len(lines) / 2
    for line in lines:
        x = (x0 + x1) / 2 - width(line, low) / 2
        for run, cjk in _runs(line):
            font = _font(cjk, low)
            draw.text((x, y + step / 2), run, fill=colour, font=font, anchor="lm")
            x += draw.textlength(run, font=font)
        y += step


def drawing(spec: dict):
    """An image drawn from a layout: ``size`` [w, h] px, ``fill`` colour and
    ``items`` in order, each with ``box`` [x0, y0, x1, y1] (fractions of the
    image) and one of: ``color`` (a rectangle; ``shape`` "ellipse" or
    "round" for others; ``outline`` width px for a ring), ``gradient`` [top,
    bottom] colours, ``text`` and ``color`` (fitted into the box), or
    ``dots`` [count, radius px, colour] (fixed pseudo-random speckles)."""
    from PIL import Image, ImageDraw

    width, height = (int(v) for v in spec["size"])
    image = Image.new("RGB", (width, height), _rgb(spec.get("fill", "#ffffff")))
    draw = ImageDraw.Draw(image)
    for item in spec.get("items", []):
        x0, y0, x1, y1 = item["box"]
        box = (x0 * width, y0 * height, x1 * width, y1 * height)
        if "text" in item:
            _text(draw, box, item["text"], _rgb(item["color"]))
        elif "gradient" in item:
            top, bottom = (np.array(_rgb(c), dtype=float) for c in item["gradient"])
            rows = max(int(box[3] - box[1]), 1)
            for k in range(rows):
                colour = tuple(
                    int(v) for v in top + (bottom - top) * k / max(rows - 1, 1)
                )
                draw.line([(box[0], box[1] + k), (box[2], box[1] + k)], fill=colour)
        elif "dots" in item:
            count, radius, colour = item["dots"]
            random = np.random.default_rng(int(count))
            for x, y in random.random((int(count), 2)):
                cx, cy = box[0] + x * (box[2] - box[0]), box[1] + y * (box[3] - box[1])
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=_rgb(colour),
                )
        else:
            shape, colour = item.get("shape", "rect"), _rgb(item["color"])
            ring = int(item.get("outline", 0))
            fill, outline = (None, colour) if ring else (colour, None)
            if shape == "ellipse":
                draw.ellipse(box, fill=fill, outline=outline, width=ring or 1)
            elif shape == "round":
                radius = min(box[2] - box[0], box[3] - box[1]) / 4
                draw.rounded_rectangle(
                    box, radius, fill=fill, outline=outline, width=ring or 1
                )
            else:
                draw.rectangle(box, fill=fill, outline=outline, width=ring or 1)
    return image


def _picture(spec: dict, cache: Path):
    """The RGB image of a paint source: a ``draw`` layout (see drawing), or
    ``url`` / ``sha256`` of an image, of an archive (``member``) or of a GLB
    (``member`` "glb:<k>", its k-th image); then ``crop`` [x0, y0, x1, y1]
    px and ``rotate`` (deg counterclockwise, by 90)."""
    from PIL import Image

    if "draw" in spec:
        return _turned(drawing(spec["draw"]), spec)
    path = download(spec["url"], spec["sha256"], cache)
    member = spec.get("member", "")
    if member.startswith("glb:"):
        data = path.read_bytes()
        length = struct.unpack_from("<I", data, 12)[0]
        gltf = json.loads(data[20 : 20 + length])
        binary = data[20 + length + 8 :]
        view = gltf["bufferViews"][gltf["images"][int(member[4:])]["bufferView"]]
        start = view.get("byteOffset", 0)
        raw = binary[start : start + view["byteLength"]]
    elif member:
        raw = _Archive(path).read(member)
    else:
        raw = path.read_bytes()
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if "crop" in spec:
        image = image.crop(tuple(int(v) for v in spec["crop"]))
    return _turned(image, spec)


def _turned(image, spec: dict):
    from PIL import Image

    turns = int(spec.get("rotate", 0)) // 90 % 4
    if turns:
        image = image.transpose(
            [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270][turns]
        )
    return image


def _label_strip(pieces, height: int, background, cache: Path) -> np.ndarray:
    """A 360 deg label (LABEL_WIDTH x height px; column k at k / width * 360
    deg, counterclockwise from +x seen from above, i.e. left to right seen
    from outside) made of pieces covering ``deg`` [from, to]: ``flat``
    images as they are, or ``photo`` images of the side seen from the middle
    of the span, cropped edge to edge."""
    from PIL import Image

    strip = np.zeros((height, LABEL_WIDTH, 3), dtype=np.uint8)
    strip[:] = background
    angles = (np.arange(LABEL_WIDTH) + 0.5) / LABEL_WIDTH * 360.0
    for piece in pieces:
        image = _picture(piece, cache).resize((LABEL_WIDTH, height), Image.LANCZOS)
        pixels = np.asarray(image)
        start, end = (float(v) for v in piece["deg"])
        offset = (angles - start) % 360.0
        span = (end - start) % 360.0 or 360.0
        columns = np.flatnonzero(offset < span)
        if piece.get("kind", "flat") == "photo":
            # Seen from outside, image right is the growing angle.
            side = np.sin(np.radians(offset[columns] - span / 2))
            x = (side / math.sin(math.radians(span / 2)) + 1) / 2
        else:
            x = offset[columns] / span
        source = np.clip((x * LABEL_WIDTH).astype(int), 0, LABEL_WIDTH - 1)
        strip[:, columns] = pixels[:, source]
    return strip


def revolve(profile, segments: int = SEGMENTS):
    """(vertices, triangles) of the surface of revolution of ``profile`` [[r,
    z], ...] about z, listed from the bottom center to the top center; wound
    outwards."""
    profile = np.asarray(profile, dtype=float)
    angles = np.linspace(0.0, 2 * math.pi, segments + 1)
    rings = [
        np.stack([r * np.cos(angles), r * np.sin(angles), np.full_like(angles, z)], 1)
        for r, z in profile
    ]
    vertices = np.concatenate(rings)
    width = segments + 1
    triangles = []
    for k in range(len(profile) - 1):
        for j in range(segments):
            a, b = k * width + j, k * width + j + 1
            triangles += [(a, b, b + width), (a, b + width, a + width)]
    triangles = np.array(triangles)
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return vertices, triangles[area > 1e-12]


def _box(size):
    """(vertices, triangles) of a box on z = 0, centered in x and y, wound outwards."""
    half = np.array(size, dtype=float) / 2
    vertices, triangles = [], []
    for name, (axis, sign) in FACE_AXES.items():
        right, up = (np.array(v, dtype=float) for v in FACE_FRAMES[name])
        normal = np.zeros(3)
        normal[axis] = sign
        base = len(vertices)
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            vertices.append((normal + a * right + b * up) * half + [0, 0, half[2]])
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return np.array(vertices), np.array(triangles)


@dataclass(frozen=True)
class MeasuredModel:
    label: str
    model: str
    shape: dict  # {"revolve": [[r, z], ...]}, {"box": [x, y, z]}, {"scan" / "profile": {...}}
    paint: dict  # "color", "side", "top", "bottom", "+x" ... "-z"
    mass: float
    size: tuple[float, float, float]
    source: str

    def entry(self) -> dict:
        return {
            "version": SCAN_BUILD_VERSION,
            "shape": self.shape,
            "paint": self.paint,
            "fonts": font_files(),
            "mass": self.mass,
            "size": list(self.size),
            "source": self.source,
        }


def scan_profile(points: np.ndarray, step: float) -> np.ndarray:
    """The revolved profile [[r, z], ...] (bottom center to top center, z from
    0) of a scan of a round object whose axis is roughly z: the axis is the
    line through the centers of slices ``step`` high, r is the median
    distance from it of each slice's points on the wall (within 80 % of the
    slice's outer radius), averaged with the neighbouring slices."""
    points = points - points.mean(axis=0)
    for _ in range(2):  # tilt the axis to z
        z = points[:, 2]
        heights = np.arange(z.min() + step, z.max() - step, step)
        slices = [np.abs(z - h) < step / 2 for h in heights]
        keep = [k for k, s in enumerate(slices) if s.sum() >= 8]
        centers = np.array([points[slices[k], :2].mean(axis=0) for k in keep])
        fit = np.polyfit(heights[keep], centers, 1)  # (2, 2): slope, offset
        axis = np.array([fit[0, 0], fit[0, 1], 1.0])
        axis /= np.linalg.norm(axis)
        v = np.cross(axis, [0.0, 0.0, 1.0])
        c = axis[2]
        k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        turn = np.eye(3) + k + k @ k / (1 + c)
        points = (points - [fit[1, 0], fit[1, 1], 0.0]) @ turn.T
    z = points[:, 2] - points[:, 2].min()
    radius = np.hypot(points[:, 0], points[:, 1])
    top = z.max()
    edges = np.arange(0.0, top + step, step)
    profile = [[0.0, 0.0]]
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (z >= low) & (z < high)
        if inside.sum() >= 8:
            # The wall: not the points of an end (a lid, a closed scan bottom).
            ring = radius[inside]
            wall = ring[ring >= 0.8 * np.percentile(ring, 95)]
            profile.append([float(np.median(wall)), float((low + high) / 2)])
    profile[1][1] = 0.0
    profile[-1][1] = float(top)
    profile.append([0.0, float(top)])
    profile = np.array(profile)
    # Scan noise: the mean of each wall radius and its neighbours.
    profile[2:-2, 0] = (profile[1:-3, 0] + profile[2:-2, 0] + profile[3:-1, 0]) / 3
    return profile


def _shape(shape: dict, cache: Path):
    """(vertices, triangles) of a measured shape, z up from z = 0, centered in x, y."""
    if "revolve" in shape:
        return revolve(shape["revolve"])
    if "box" in shape:
        return _box(shape["box"])
    spec = shape.get("scan") or shape["profile"]
    mesh = read_scan(download(spec["url"], spec["sha256"], cache), spec.get("mesh", ""))
    turn = _rpy_matrix(*(math.radians(float(v)) for v in spec.get("rpy", (0, 0, 0))))
    vertices = mesh.vertices @ turn.T * float(spec.get("scale", 1.0))
    if "profile" in shape:
        return revolve(scan_profile(vertices, float(spec.get("step", 0.002))))
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    vertices -= [(lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2, lower[2]]
    triangles = mesh.corners[:, :, 0] - 1
    a, b, c = (vertices[triangles[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:
        triangles = triangles[:, [0, 2, 1]]
    return vertices, triangles


def build_measured(model: MeasuredModel, cache: Path) -> Path:
    """Build (once) a model of measured shape painted with images.

    Shapes (z up, bottom at 0): a revolved profile ``[[r, z], ...]`` (m,
    bottom center to top center), a ``box`` [x, y, z], a metric ``scan``
    (url, sha256, mesh, rpy deg, scale) as it is, or the ``profile`` of a
    round object measured from such a scan (see scan_profile). Paint: ``side`` is a label around z
    between ``z`` [z0, z1] made of ``pieces`` (see _label_strip), ``radius``
    sets its pixel aspect; ``top`` / ``bottom`` are images of the end discs
    (``radius`` m) seen from outside with +y / -y up; ``+x`` ... ``-z``
    cover the box faces (see FACE_FRAMES); the rest has ``color``."""
    from PIL import Image

    directory = cache / model.model
    if _built(directory, model.entry()):
        return directory
    vertices, triangles = _shape(model.shape, cache)
    paint = model.paint
    background = _rgb(paint.get("color", "#ffffff"))
    corner = vertices[triangles]
    normals = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centroid = corner.mean(axis=1)
    # Texture atlas: a band per painted part (images at its left), then a
    # swatch of the background colour.
    bands, widths, parts = [], [], {}
    if "side" in paint:
        side = paint["side"]
        z0, z1 = (float(v) for v in side["z"])
        circumference = 2 * math.pi * float(side["radius"])
        rows = max(int(LABEL_WIDTH * (z1 - z0) / circumference), 8)
        parts["side"] = len(bands)
        bands.append(_label_strip(side["pieces"], rows, background, cache))
        widths.append(LABEL_WIDTH)
    for name in ("top", "bottom", *FACE_AXES):
        if name in paint:
            image = _picture(paint[name], cache)
            scale = min(1.0, LABEL_WIDTH / image.width)
            size = (max(int(image.width * scale), 1), max(int(image.height * scale), 1))
            image = image.resize(size, Image.LANCZOS)
            band = np.zeros((image.height, LABEL_WIDTH, 3), dtype=np.uint8)
            band[:] = background
            band[:, : image.width] = np.asarray(image)
            parts[name] = len(bands)
            bands.append(band)
            widths.append(image.width)
    swatch = np.zeros((8, LABEL_WIDTH, 3), dtype=np.uint8)
    swatch[:] = background
    atlas = np.concatenate([*bands, swatch])
    total = atlas.shape[0]
    tops = np.cumsum([0, *[band.shape[0] for band in bands]])

    def band_uv(index, x, y):
        """Atlas uv of (x, y) in [0, 1] (y up) of the image of band ``index``."""
        rows = bands[index].shape[0]
        u = x * widths[index] / LABEL_WIDTH
        v = 1.0 - (tops[index] + (1.0 - np.clip(y, 0, 1)) * (rows - 1) + 0.5) / total
        return np.stack([u, v], axis=-1)

    uvs = np.zeros((len(triangles), 3, 2))
    uvs[:] = [0.5, 4.0 / total]  # the swatch
    if "side" in parts:
        z0, z1 = (float(v) for v in paint["side"]["z"])
        on = (
            (np.abs(normals[:, 2]) < 0.8)
            & (centroid[:, 2] >= z0)
            & (centroid[:, 2] <= z1)
        )
        theta = np.degrees(np.arctan2(corner[..., 1], corner[..., 0])) % 360
        theta = theta[:, :1] + (theta - theta[:, :1] + 180) % 360 - 180  # seam
        x = theta / 360.0
        y = (corner[..., 2] - z0) / (z1 - z0)
        uvs[on] = band_uv(parts["side"], x[on], y[on])
    for name, sign in (("top", 1), ("bottom", -1)):
        if name in parts:
            radius = float(paint[name]["radius"])
            on = (normals[:, 2] * sign > 0.8) & (
                np.hypot(centroid[:, 0], centroid[:, 1]) <= radius
            )
            x = corner[..., 0] / (2 * radius) + 0.5
            y = sign * corner[..., 1] / (2 * radius) + 0.5
            uvs[on] = band_uv(parts[name], x[on], y[on])
    middle = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    for name, (axis, sign) in FACE_AXES.items():
        if name in parts:
            on = normals[:, axis] * sign > 0.8
            right, up = (np.array(v, dtype=float) for v in FACE_FRAMES[name])
            x = (corner - middle) @ right / (extent @ np.abs(right)) + 0.5
            y = (corner - middle) @ up / (extent @ np.abs(up)) + 0.5
            uvs[on] = band_uv(parts[name], x[on], y[on])
    count = len(triangles)
    index = np.arange(1, 3 * count + 1).reshape(count, 3)
    image = io.BytesIO()
    Image.fromarray(atlas).save(image, "PNG")
    # Shared vertices (smooth normals), a uv per corner.
    mesh = _mesh(
        vertices,
        uvs.reshape(-1, 2),
        np.zeros((0, 3)),
        np.stack([triangles + 1, index, np.zeros_like(index)], axis=2),
        np.zeros(count),
        [{"texture": image.getvalue(), "color": (1.0, 1.0, 1.0)}],
    )
    _finish(mesh, model.size, model.label)
    (directory / "build.json").unlink(missing_ok=True)
    _write_model(directory, model.model, mesh, model.mass, model.source)
    _stamp(directory, model.entry())
    return directory

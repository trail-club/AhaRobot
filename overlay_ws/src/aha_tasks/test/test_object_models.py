"""Models of real objects built from datasets: readers, scans, measured shapes."""

from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import posixpath
import struct
import sys
import tarfile
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aha_tasks.object_models import (  # noqa: E402
    SCAN_TEXTURE_SIZE,
    MeasuredModel,
    ModelError,
    ScannedModel,
    build_measured,
    build_scanned,
    read_glb,
    read_obj,
    scan_profile,
    solid_inertia,
)
from aha_tasks.sim_objects import YcbModel, ycb_bounds, ycb_sdf  # noqa: E402


BOX = np.array([[x, y, z] for x in (-1, 1) for y in (-0.5, 0.5) for z in (0.0, 0.5)])
# Outward quads of BOX; the last one (x = +1) is the lid.
QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4)]
LID = (1, 5, 7, 3)


def png(size, colour=(200, 30, 30)) -> bytes:
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", size, colour).save(data, "PNG")
    return data.getvalue()


def write_archive(path: Path, files: dict) -> Path:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "w") as out:
            for name, data in files.items():
                out.writestr(name, data)
    else:
        with tarfile.open(path, "w:gz") as out:
            for name, data in files.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                out.addfile(info, io.BytesIO(data))
    return path


def box_scan(directory: Path, kind: str) -> Path:
    """A 2 x 1 x 0.5 (scan units) box OBJ like the YCB Berkeley meshes: wound
    inwards, normals that do not exist, the texture in another folder; an
    untextured lid, and a board under the box (a turntable) to crop away."""
    board = BOX * [2, 2, 0] + [0, 0, -1]
    lines = ["mtllib scan.mtl", "usemtl a"]
    lines += [f"v {x} {y} {z}" for x, y, z in np.concatenate([BOX, board])]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "vn 0 0 1"]
    for quad in [*QUADS, LID, (8, 10, 14, 12)]:  # written inwards, normal 9
        if quad == LID:
            lines.append("usemtl lid")
        a, b, c, d = (k + 1 for k in quad)
        lines.append(f"f {d}/4/9 {c}/3/9 {b}/2/9 {a}/1/9")
    mtl = b"newmtl a\nKd 0.6 0.6 0.6\nmap_Kd scan.png\n\nnewmtl lid\nKd 0.2 0.4 0.6\n"
    files = {
        "obj/poisson/scan.obj": "\n".join(lines).encode(),
        "obj/poisson/scan.mtl": mtl,
        "obj/textures/scan.png": png((3000, 100)),
    }
    return write_archive(directory / f"scan.{kind}", files)


def box_glb(path: Path) -> Path:
    """BOX as a binary glTF: y up, scaled by 2 in a node moved by 5 along z."""
    points = (BOX[:, [0, 2, 1]] * [1, 1, -1]).astype("<f4")  # z up -> y up
    uv = np.tile([[0.0, 0.0], [1.0, 1.0]], (4, 1)).astype("<f4")
    triangles = []
    for a, b, c, d in [*QUADS, LID]:
        triangles += [(a, b, c), (a, c, d)]
    indices = np.array(triangles, dtype="<u2")
    binary = points.tobytes() + uv.tobytes() + indices.tobytes()
    image = png((8, 8))
    binary += image + b"\0" * (-len(binary + image) % 4)
    sizes = [points.nbytes, uv.nbytes, indices.nbytes, len(image)]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()
    gltf = {
        "asset": {"version": "2.0"},
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"children": [1], "translation": [0, 0, 5]},
            {"mesh": 0, "scale": [2, 2, 2]},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "indices": 2,
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}],
        "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": o, "byteLength": n}
            for o, n in zip(offsets, sizes)
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC2"},
            {
                "bufferView": 2,
                "componentType": 5123,
                "count": len(indices) * 3,
                "type": "SCALAR",
            },
        ],
    }
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    body = struct.pack("<I4s", len(text), b"JSON") + text
    body += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body)
    return path


def built_mesh(directory: Path, name: str):
    """The built OBJ of a model directory, read back."""

    class Folder:
        def read(self, member, near=""):
            return (directory / "meshes" / posixpath.basename(member)).read_bytes()

    return read_obj(Folder(), f"{name}.obj")


class ScannedModelTests(unittest.TestCase):
    def scanned(self, archive, **changes):
        fields = dict(
            label="box",
            model="test_box",
            url=archive.as_uri(),
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            mesh="obj/poisson/scan.obj",
            rpy=(0.0, 0.0, np.pi / 2),
            scale=0.1,
            crop=((-2.0, -2.0, -0.1), (2.0, 2.0, 1.0)),
            mass=0.5,
            size=(0.1, 0.2, 0.05),
            source="test",
        )
        fields.update(changes)
        return ScannedModel(**fields)

    def test_build_turns_crops_centers_and_orients_the_scan(self):
        for kind in ("zip", "tgz"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                model = self.scanned(box_scan(tmp, kind))
                directory = build_scanned(model, tmp / "cache")
                center, size = ycb_bounds(directory, 0.0)
                np.testing.assert_allclose(center, 0, atol=1e-6)
                np.testing.assert_allclose(size, [0.1, 0.2, 0.05], atol=1e-6)
                mesh = built_mesh(directory, "test_box")
                self.assertEqual(len(mesh.vertices), 8)  # the board is cropped
                v, corners = mesh.vertices, mesh.corners
                a, b, c = (v[corners[:, k, 0] - 1] for k in range(3))
                faces = np.cross(b - a, c - a)
                self.assertTrue(np.all(np.einsum("ij,ij->i", faces, a) > 0))
                normals = mesh.normals[corners[:, :, 2] - 1].sum(axis=1)
                self.assertTrue(np.all(np.einsum("ij,ij->i", faces, normals) > 0))
                # The textured sides and the plain lid, named after the model.
                self.assertEqual(len(mesh.materials), 2)
                self.assertIsNotNone(mesh.materials[0]["texture"])
                self.assertEqual(mesh.materials[1]["texture"], None)
                np.testing.assert_allclose(mesh.materials[1]["color"], (0.2, 0.4, 0.6))
                mtl = (directory / "meshes" / "test_box.mtl").read_text()
                self.assertIn("newmtl test_box_0", mtl)
                self.assertIn("map_Kd test_box_0.png", mtl)
                from PIL import Image

                with Image.open(directory / "meshes" / "test_box_0.png") as image:
                    self.assertEqual(max(image.size), SCAN_TEXTURE_SIZE)
                inertial = ET.parse(directory / "model-1_4.sdf").find(
                    "model/link/inertial"
                )
                self.assertEqual(float(inertial.findtext("mass")), 0.5)
                ixx = float(inertial.findtext("inertia/ixx"))
                self.assertAlmostEqual(ixx, 0.5 * (0.2**2 + 0.05**2) / 12)
                # Built once; the SDF points at the built files.
                stamp = (directory / "build.json").stat().st_mtime_ns
                self.assertEqual(build_scanned(model, tmp / "cache"), directory)
                self.assertEqual((directory / "build.json").stat().st_mtime_ns, stamp)
                # A changed size is checked again, not taken from the cache.
                with self.assertRaises(ModelError):
                    build_scanned(replace(model, size=(0.1, 0.2, 0.06)), tmp / "cache")
                built = YcbModel("box", "test_box", 0.0, directory)
                sdf = ET.fromstring(ycb_sdf(built, "box_1", (0, 0, 0.1, 0)))
                uri = sdf.findtext("model/link/visual/geometry/mesh/uri")
                self.assertEqual(uri, (directory / "meshes" / "test_box.obj").as_uri())

    def test_glb_nodes_and_y_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = box_glb(Path(tmp) / "box.glb")
            mesh = read_glb(path.read_bytes())
            np.testing.assert_allclose(mesh.vertices.min(0), [-2, 0, 4], atol=1e-6)
            np.testing.assert_allclose(mesh.vertices.max(0), [2, 1, 6], atol=1e-6)
            self.assertEqual(sorted(np.unique(mesh.uvs[:, 1])), [0.0, 1.0])
            self.assertEqual(len(mesh.corners), 12)
            self.assertIsNotNone(mesh.materials[0]["texture"])
            # Turned z up (+90 deg about x) into the model.
            model = self.scanned(
                path,
                mesh="",
                rpy=(np.pi / 2, 0.0, 0.0),
                crop=None,
                scale=0.05,
                size=(0.2, 0.1, 0.05),
            )
            directory = build_scanned(model, Path(tmp) / "cache")
            np.testing.assert_allclose(ycb_bounds(directory, 0.0)[1], [0.2, 0.1, 0.05])

    def test_ply_with_vertex_texture_coordinates(self):
        lines = [
            "ply",
            "format ascii 1.0",
            "comment TextureFile t.png",
            "element vertex 8",
            "property float x",
            "property float y",
            "property float z",
            "property float texture_u",
            "property float texture_v",
            "element face 6",
            "property list uchar int vertex_indices",
            "end_header",
        ]
        lines += [f"{x * 10} {y * 10} {z * 10} 0.5 0.5" for x, y, z in BOX]
        lines += ["4 " + " ".join(str(k) for k in q) for q in [*QUADS, LID]]
        with tempfile.TemporaryDirectory() as tmp:
            archive = write_archive(
                Path(tmp) / "m.zip",
                {
                    "models/obj_1.ply": "\n".join(lines).encode(),
                    "models/t.png": png((4, 4)),
                },
            )
            model = self.scanned(
                archive, mesh="models/obj_1.ply", scale=0.01, crop=None
            )
            directory = build_scanned(model, Path(tmp) / "cache")
            np.testing.assert_allclose(ycb_bounds(directory, 0.0)[1], [0.1, 0.2, 0.05])

    def test_build_refuses_a_changed_download_or_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            archive = box_scan(tmp, "zip")
            with self.assertRaises(ModelError):
                build_scanned(self.scanned(archive, sha256="0" * 64), tmp / "cache")
            with self.assertRaises(ModelError):
                build_scanned(self.scanned(archive, size=(0.2, 0.1, 0.05)), tmp / "c")
            with self.assertRaises(ModelError):  # the board left in
                build_scanned(self.scanned(archive, crop=None), tmp / "c")

    def test_a_cut_download_is_a_model_error(self):
        import http.client
        from unittest import mock

        from aha_tasks.object_models import download

        reply = mock.MagicMock()
        reply.__enter__.return_value.read.side_effect = http.client.IncompleteRead(b"")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "urllib.request.urlopen", return_value=reply
        ):
            with self.assertRaises(ModelError):
                download("https://example.com/scan.zip", "0" * 64, Path(tmp))

    def test_solid_inertia_of_a_cube(self):
        points = np.array(
            [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float
        )
        from aha_tasks.object_models import _outward_hull

        center, inertia = solid_inertia(points, _outward_hull(points), 6.0)
        np.testing.assert_allclose(center, [0.5, 0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(inertia, np.eye(3), atol=1e-12)


def quarters(path: Path) -> Path:
    """A 4 x 1 px label: red, green, blue, white quarters (left to right)."""
    from PIL import Image

    image = Image.new("RGB", (4, 1))
    for k, colour in enumerate(
        [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    ):
        image.putpixel((k, 0), colour)
    image.save(path)
    return path


def source(path: Path, **extra) -> dict:
    return dict(
        url=path.as_uri(), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **extra
    )


def colour_at(directory: Path, name: str, point) -> tuple:
    """Texture colour of the built mesh's triangle nearest ``point``, at its center."""
    from PIL import Image

    mesh = built_mesh(directory, name)
    centers = mesh.vertices[mesh.corners[:, :, 0] - 1].mean(axis=1)
    k = np.argmin(np.linalg.norm(centers - point, axis=1))
    u, v = mesh.uvs[mesh.corners[k, :, 1] - 1].mean(axis=0)
    with Image.open(directory / "meshes" / f"{name}_0.png") as image:
        image = image.convert("RGB")
        x = int(np.clip(u % 1.0, 0, 0.9999) * image.width)
        y = int(np.clip(1.0 - v, 0, 0.9999) * image.height)
        return image.getpixel((x, y))


class MeasuredModelTests(unittest.TestCase):
    def test_revolved_label_wraps_counterclockwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            label = quarters(tmp / "label.png")
            model = MeasuredModel(
                "can",
                "test_can",
                {"revolve": [[0, 0], [0.03, 0], [0.03, 0.1], [0, 0.1]]},
                {
                    "color": "#000000",
                    "side": {
                        "z": [0, 0.1],
                        "radius": 0.03,
                        "pieces": [source(label, deg=[0, 360])],
                    },
                },
                0.1,
                (0.06, 0.06, 0.1),
                "test",
            )
            directory = build_measured(model, tmp / "cache")
            np.testing.assert_allclose(
                ycb_bounds(directory, 0.0)[1], [0.06, 0.06, 0.1], atol=1e-3
            )
            # Left to right seen from outside is counterclockwise from +x:
            # red at 45 deg, green at 135, blue at 225, white at 315; black caps.
            colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
            for k, colour in enumerate(colours):
                angle = np.radians(45 + 90 * k)
                point = (0.03 * np.cos(angle), 0.03 * np.sin(angle), 0)
                with self.subTest(angle=45 + 90 * k):
                    np.testing.assert_allclose(
                        colour_at(directory, "test_can", point), colour, atol=16
                    )
            self.assertEqual(colour_at(directory, "test_can", (0, 0, 0.05)), (0, 0, 0))

    def test_box_faces_and_profile_from_a_tilted_scan(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            front = tmp / "front.png"
            Image.new("RGB", (8, 4), (0, 0, 255)).save(front)
            model = MeasuredModel(
                "tab",
                "test_tab",
                {"box": [0.04, 0.03, 0.015]},
                {"color": "#ffffff", "+z": source(front)},
                0.02,
                (0.04, 0.03, 0.015),
                "test",
            )
            directory = build_measured(model, tmp / "cache")
            np.testing.assert_allclose(
                colour_at(directory, "test_tab", (0, 0, 0.0075)), (0, 0, 255), atol=16
            )
            self.assertEqual(
                colour_at(directory, "test_tab", (0.02, 0, 0)), (255, 255, 255)
            )
            # A cylinder of radius 2 and height 10 scanned 3 deg off its axis.
            angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
            heights = np.linspace(0, 10, 101)
            points = np.array(
                [[2 * np.cos(a), 2 * np.sin(a), h] for h in heights for a in angles]
            )
            tilt = np.radians(3)
            turn = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(tilt), -np.sin(tilt)],
                    [0, np.sin(tilt), np.cos(tilt)],
                ]
            )
            profile = scan_profile(points @ turn.T + [5, -1, 2], 0.2)
            np.testing.assert_allclose(profile[1:-1, 0], 2, atol=0.02)
            self.assertAlmostEqual(profile[-1, 1], 10, delta=0.15)


if __name__ == "__main__":
    unittest.main()

"""Build the Waveshare case cradle in Fusion. Run as a Fusion Python script.

All dimensions are mm. The assembly uses the upstream Astra STEP coordinate
system (+Y up, +Z forward). Each component has its own local origin.
The model is regenerated in a NEW document; existing documents are untouched.
"""

import json
import struct
from pathlib import Path

import adsk.core as core
import adsk.fusion as fusion

OUT = Path(__file__).resolve().parent.parent
# Verified case envelope from Waveshare's with-case PDF. Not VESA dimensions.
CASE_W, CASE_H, CASE_D = 254.0, 187.0, 26.1
SCREEN_BOTTOM, SCREEN_BACK = 1000.0, 560.0
BEAM_UNDERSIDE, BEAM_BACK = 1341.0, 425.26820827052
MAST_BOTTOM, MAST_TOP = 993.0, 1233.0
MAST_BACK, MAST_FRONT = 490.0, 510.0
SIDE_GAP, REAR_PAD, FRONT_PAD = 1.3, 2.0, 1.0
BOTTOM_PAD, TOP_PAD = 2.0, 1.0
NEW = fusion.FeatureOperations.NewBodyFeatureOperation
CUT = fusion.FeatureOperations.CutFeatureOperation


def point(x, y, z):
    return core.Point3D.create(x / 10, y / 10, z / 10)


def collection(items):
    c = core.ObjectCollection.create()
    for item in items:
        c.add(item)
    return c


class Part:
    def __init__(self, root, name, origin):
        self.name, self.origin = name, origin
        t = core.Matrix3D.create()
        t.translation = core.Vector3D.create(*[v / 10 for v in origin])
        self.occ = root.occurrences.addNewComponent(t)
        self.comp = self.occ.component
        self.comp.name = name
        self.comp.partNumber = name

    def local(self, xyz):
        return point(*[xyz[i] - self.origin[i] for i in range(3)])

    def sketch(self, axis, start, name):
        planes = (
            self.comp.yZConstructionPlane,
            self.comp.xZConstructionPlane,
            self.comp.xYConstructionPlane,
        )
        base = planes[axis]
        sign = base.geometry.normal.asArray()[axis]
        offset = (start - self.origin[axis]) / sign / 10
        inp = self.comp.constructionPlanes.createInput()
        inp.setByOffset(base, core.ValueInput.createByReal(offset))
        plane = self.comp.constructionPlanes.add(inp)
        plane.name = name + "_plane"
        sketch = self.comp.sketches.add(plane)
        sketch.name = name + "_profile"
        plane.isLightBulbOn = False
        return sketch, sign

    def poly(self, axis, start, end, vertices, name, operation=NEW):
        sketch, sign = self.sketch(axis, start, name)
        axes = [i for i in range(3) if i != axis]
        pts = []
        for a, b in vertices:
            xyz = [0.0, 0.0, 0.0]
            xyz[axis], xyz[axes[0]], xyz[axes[1]] = start, a, b
            pts.append(sketch.modelToSketchSpace(self.local(xyz)))
        for p, q in zip(pts, pts[1:] + pts[:1]):
            line = sketch.sketchCurves.sketchLines.addByTwoPoints(p, q)
            line.isFixed = True
        assert sketch.profiles.count == 1, name
        feature = self.comp.features.extrudeFeatures.addSimple(
            sketch.profiles.item(0),
            core.ValueInput.createByReal((end - start) / sign / 10),
            operation,
        )
        feature.name = name
        sketch.isVisible = False
        return feature

    def box(self, lo, hi, name):
        x, y, z = lo
        X, Y, Z = hi
        return self.poly(2, z, Z, [(x, y), (X, y), (X, Y), (x, Y)], name)

    def holes(self, axis, start, end, centers, diameter, name):
        sk, sign = self.sketch(axis, start, name)
        axes = [i for i in range(3) if i != axis]
        for a, b in centers:
            xyz = [0.0, 0.0, 0.0]
            xyz[axis], xyz[axes[0]], xyz[axes[1]] = start, a, b
            circle = sk.sketchCurves.sketchCircles.addByCenterRadius(
                sk.modelToSketchSpace(self.local(xyz)), diameter / 20
            )
            circle.isFixed = True
        feature = self.comp.features.extrudeFeatures.addSimple(
            collection(list(sk.profiles)),
            core.ValueInput.createByReal((end - start) / sign / 10),
            CUT,
        )
        feature.name = name
        sk.isVisible = False

    def join(self):
        bodies = list(self.comp.bRepBodies)
        if len(bodies) > 1:
            ci = self.comp.features.combineFeatures.createInput(
                bodies[0], collection(bodies[1:])
            )
            ci.operation = fusion.FeatureOperations.JoinFeatureOperation
            ci.isKeepToolBodies = False
            f = self.comp.features.combineFeatures.add(ci)
            f.name = self.name + "_join"
        assert self.comp.bRepBodies.count == 1, self.name
        b = self.comp.bRepBodies.item(0)
        assert b.isSolid and b.lumps.count == 1, self.name
        b.name = self.name
        return b


def make_adapter(root):
    # The rear leg goes behind the pan servo's swept envelope. The dogleg
    # only reaches forward below the head, where the 2020 mast starts.
    p = Part(root, "01_frame_adapter", [-28, 1180, BEAM_BACK])
    p.box((-28, 1333, BEAM_BACK), (28, 1341, 485), "beam_foot_8mm")
    p.box((-25, 1230, 434), (25, 1333.2, 440), "rear_drop_6mm")
    p.box((-25, 1224, 434), (25, 1232, 490), "dogleg_bridge_8mm")
    p.box((-25, 1180, 484), (25, 1232, 490), "mast_rear_plate_6mm")
    for a, b in [(-25, -19), (19, 25)]:
        p.poly(
            0, a, b, [(1232, 440), (1250, 440), (1232, 484)], "dogleg_gusset_" + str(a)
        )
    p.join()
    p.holes(
        1,
        1332.9,
        1341.1,
        [(x, z) for x in (-15, 15) for z in (450.26820827052, 475.26820827052)],
        6.6,
        "four_M6_frame_holes",
    )
    p.holes(2, 483.9, 490.1, [(0, 1195), (0, 1220)], 5.5, "two_M5_mast_holes")
    return p


def make_lower(root, side):
    left = side < 0
    name = "02_lower_left" if left else "03_lower_right"
    p = Part(root, name, [-133.3 if left else -18, 1012, MAST_FRONT])
    low, high = (1075, 1117) if left else (1027, 1069)
    # Disjoint central bolt lands; each half has two M5 bolts into the front slot.
    p.box((-18, low, 510), (18, high, 518), "mast_bolt_land_8mm")
    verts = [(18, low), (95, 1018), (128.3, 1018), (128.3, 1080), (18, high)]
    if left:
        verts = [(-x, y) for x, y in verts][::-1]
    p.poly(2, 510, 518, verts, "diagonal_back_arm_8mm")

    def bx(x1, y1, z1, x2, y2, z2, label):
        a, b = sorted((side * x1, side * x2))
        p.box((a, y1, z1), (b, y2, z2), label)

    bx(95, 1012, 510, 133.3, 1018, 592.1, "bottom_shelf_6mm")
    bx(95, 1018, 552, 133.3, 1080, 558, "rear_pad_wall_6mm")
    bx(128.3, 1018, 558, 133.3, 1080, 592.1, "side_stop_5mm")
    bx(95, 1018, 587.1, 133.3, 1027, 592.1, "front_bezel_lip_5mm")
    for x1, x2 in [(98, 104), (122, 128)]:
        a, b = sorted((side * x1, side * x2))
        p.poly(
            0, a, b, [(1018, 518), (1080, 518), (1018, 558)], "shelf_gusset_" + str(x1)
        )
    # End wall bridges between the two gussets; slice with supports under this wall.
    p.join()
    p.holes(2, 509.9, 518.1, [(0, low + 10), (0, high - 10)], 5.5, "two_M5_mast_holes")
    p.holes(2, 516, 518.1, [(0, low + 10), (0, high - 10)], 11, "M5_washer_recess_2mm")
    return p


def make_top(root):
    p = Part(root, "04_top_keeper", [-45, 1152, 510])
    p.box((-18, 1152, 510), (18, 1208.2, 518), "mast_land_8mm")
    # 1 mm uncompressed pad above the 1207 mm case top.
    p.box((-45, 1208, 510), (45, 1214, 592.1), "top_shelf_6mm")
    p.box((-45, 1201, 587.1), (45, 1208.2, 592.1), "front_lip_5mm")
    p.box((-45, 1190, 552), (45, 1208.2, 558), "upper_rear_pad_wall_6mm")
    for x1, x2 in [(-18, -12), (12, 18)]:
        p.poly(
            0,
            x1,
            x2,
            [(1165, 518), (1208, 518), (1208, 556)],
            "keeper_gusset_" + str(x1),
        )
    p.join()
    p.holes(2, 509.9, 518.1, [(0, 1162), (0, 1187)], 5.5, "two_M5_mast_holes")
    p.holes(2, 516, 518.1, [(0, 1162), (0, 1187)], 11, "M5_washer_recess_2mm")
    # Two separate tie slots; cable strain relief/tether can loop through here.
    for x in (-34, 34):
        p.poly(
            1,
            1207.9,
            1214.1,
            [(x - 2, 532), (x + 2, 532), (x + 2, 544), (x - 2, 544)],
            "strap_slot_" + str(x),
            CUT,
        )
    return p


def view(app, eye, target, name):
    cam = app.activeViewport.camera
    cam.cameraType = core.CameraTypes.OrthographicCameraType
    cam.eye = point(*eye)
    cam.target = point(*target)
    cam.upVector = core.Vector3D.create(0, 1, 0)
    cam.isFitView = True
    app.activeViewport.camera = cam
    app.activeViewport.saveAsImageFile(str(OUT / name), 1500, 1200)


def run(context):
    OUT.mkdir(parents=True, exist_ok=True)
    app = core.Application.get()
    doc = app.documents.add(core.DocumentTypes.FusionDesignDocumentType)
    doc.name = "AhaRobot_Waveshare_display_mount"
    d = fusion.Design.cast(app.activeProduct)
    d.designType = fusion.DesignTypes.ParametricDesignType
    d.fusionUnitsManager.distanceDisplayUnits = (
        fusion.DistanceUnits.MillimeterDistanceUnits
    )
    root = d.rootComponent
    parts = [
        make_adapter(root),
        make_lower(root, -1),
        make_lower(root, 1),
        make_top(root),
    ]
    for p in parts[1:]:
        delta = SCREEN_BOTTOM - 1020.0
        t = p.occ.transform2
        v = t.translation
        v.y += delta / 10
        t.translation = v
        p.occ.transform2 = t
        p.origin[1] += delta
    if d.snapshots.hasPendingSnapshot:
        d.snapshots.add()
    # Round the exposed outer edges, leaving all bearing faces flat.
    for p in parts:
        b = p.comp.bRepBodies.item(0)
        bb = b.boundingBox
        hi = bb.maxPoint.asArray()
        edges = []
        for edge in b.edges:
            if not core.Line3D.cast(edge.geometry):
                continue
            a, z = (
                edge.startVertex.geometry.asArray(),
                edge.endVertex.geometry.asArray(),
            )
            if p.name.startswith("01"):
                on_x = abs(a[0]) < 1e-6 or abs(a[0] - hi[0]) < 1e-6
                on_z = abs(a[2]) < 1e-6 or abs(a[2] - hi[2]) < 1e-6
                take = (
                    on_x
                    and on_z
                    and abs(a[0] - z[0]) < 1e-6
                    and abs(a[2] - z[2]) < 1e-6
                )
            else:
                # Vertical outside corner where a person may touch the cradle.
                outside_x = 0 if p.name.startswith("02") else hi[0]
                take = (
                    abs(a[0] - outside_x) < 1e-6
                    and abs(a[2] - hi[2]) < 1e-6
                    and abs(a[0] - z[0]) < 1e-6
                    and abs(a[2] - z[2]) < 1e-6
                )
            if take:
                edges.append(edge)
        if edges:
            fi = p.comp.features.filletFeatures.createInput()
            fi.edgeSetInputs.addConstantRadiusEdgeSet(
                collection(edges), core.ValueInput.createByReal(0.1), False
            )
            p.comp.features.filletFeatures.add(fi).name = "exposed_corner_R1"
    # Reference solids are envelopes, not manufacturing geometry.
    mast = Part(root, "REF_2020_L240_envelope", [-10, MAST_BOTTOM, MAST_BACK])
    mast.box(
        (-10, MAST_BOTTOM, MAST_BACK), (10, MAST_TOP, MAST_FRONT), "stock_envelope"
    )
    case = Part(root, "REF_Waveshare_case_envelope", [-127, SCREEN_BOTTOM, SCREEN_BACK])
    case.box(
        (-127, SCREEN_BOTTOM, SCREEN_BACK),
        (127, SCREEN_BOTTOM + CASE_H, SCREEN_BACK + CASE_D),
        "case_envelope",
    )
    # Separate active-area reference on the front; not included in exports for printing.
    active = Part(root, "REF_active_area", [-108.95, SCREEN_BOTTOM + 25.2, 586.1])
    active.box(
        (-108.95, SCREEN_BOTTOM + 25.2, 586.1),
        (108.95, SCREEN_BOTTOM + 161.8, 586.2),
        "active_area",
    )
    for part, rgb in [(p, (40, 120, 180)) for p in parts] + [
        (mast, (155, 160, 167)),
        (case, (35, 38, 43)),
        (active, (175, 212, 232)),
    ]:
        b = part.comp.bRepBodies.item(0)
        appearance = d.appearances.addByCopy(b.appearance, part.name + "_color")
        for prop_id in ("opaque_albedo", "surface_albedo"):
            color = core.ColorProperty.cast(
                appearance.appearanceProperties.itemById(prop_id)
            )
            if color:
                color.value = core.Color.create(*rgb, 255)
        b.appearance = appearance
    report = {
        "coordinate_system": "+X as Astra STEP, +Y up, +Z forward; mm",
        "case_mm": [CASE_W, CASE_H, CASE_D],
        "case_bbox_mm": [
            [-127, SCREEN_BOTTOM, SCREEN_BACK],
            [127, SCREEN_BOTTOM + CASE_H, SCREEN_BACK + CASE_D],
        ],
        "display_vertical_offset_mm": SCREEN_BOTTOM - 1020.0,
        "mast_envelope_mm": [[-10, MAST_BOTTOM, MAST_BACK], [10, MAST_TOP, MAST_FRONT]],
        "parts": [],
        "timeline_errors": [],
    }
    em = d.exportManager
    for p in parts:
        b = p.comp.bRepBodies.item(0)
        bb = b.boundingBox
        size = [
            (bb.maxPoint.asArray()[i] - bb.minPoint.asArray()[i]) * 10 for i in range(3)
        ]
        report["parts"].append(
            {
                "name": p.name,
                "origin_mm": p.origin,
                "size_mm": size,
                "volume_mm3": b.volume * 1000,
                "solid": b.isSolid,
                "lumps": b.lumps.count,
            }
        )
        assert max(size) < 180, (p.name, size)
        assert em.execute(
            em.createSTEPExportOptions(str(OUT / (p.name + ".step")), p.comp)
        )
        st = em.createSTLExportOptions(p.comp, str(OUT / (p.name + ".stl")))
        st.unitType = fusion.DistanceUnits.MillimeterDistanceUnits
        st.meshRefinement = fusion.MeshRefinementSettings.MeshRefinementHigh
        st.isBinaryFormat = True
        assert em.execute(st)
        if p.name.startswith("01"):
            # Rigid rotation only: put the frame bearing face on the print bed.
            path = OUT / (p.name + ".stl")
            data = bytearray(path.read_bytes())
            n = struct.unpack_from("<I", data, 80)[0]
            for j in range(n):
                pos = 84 + 50 * j
                v = list(struct.unpack_from("<12f", data, pos))
                v[0:3] = [v[0], v[2], -v[1]]
                for k in (3, 6, 9):
                    x, y, z = v[k : k + 3]
                    v[k : k + 3] = [x, z, size[1] - y]
                struct.pack_into("<12f", data, pos, *v)
            path.write_bytes(data)
    for i in range(d.timeline.count):
        item = d.timeline.item(i)
        if item.healthState != fusion.FeatureHealthStates.HealthyFeatureHealthState:
            report["timeline_errors"].append(
                {"name": item.name, "message": item.errorOrWarningMessage}
            )
    assert not report["timeline_errors"], report["timeline_errors"]
    assert em.execute(
        em.createSTEPExportOptions(str(OUT / "display_mount_assembly.step"))
    )

    (OUT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    active.occ.isLightBulbOn = False
    case.occ.isLightBulbOn = False
    view(app, (550, 1400, 950), (0, 1170, 510), "mount_rear.png")
    case.occ.isLightBulbOn = True
    active.occ.isLightBulbOn = True
    view(app, (550, 1400, 1100), (0, 1170, 510), "mount_front.png")
    assert em.execute(
        em.createFusionArchiveExportOptions(str(OUT / "display_mount_assembly.f3d"))
    )
    for path in OUT.glob("*.step"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    print(json.dumps(report, indent=2))

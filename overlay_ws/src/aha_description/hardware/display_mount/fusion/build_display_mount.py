"""Build the Waveshare case cradle in Fusion. Run as a Fusion Python script.

All dimensions are mm. The assembly uses the upstream Astra STEP coordinate
system (+Y up, +Z forward). Each component has its own local origin.
Every part is printed: each fits the 180 mm cube of a Bambu Lab A1 mini, and
a printed spine stands where a 2020 profile would. The parts are joined with
M5 bolts into hex nut pockets in the spine.
The model is regenerated in a NEW document; existing documents are untouched.
"""

import json
import math
import struct
from pathlib import Path

import adsk.core as core
import adsk.fusion as fusion

OUT = Path(__file__).resolve().parent.parent
# Verified case envelope from Waveshare's with-case PDF. Not VESA dimensions.
CASE_W, CASE_H, CASE_D = 254.0, 187.0, 26.1
SCREEN_BOTTOM, SCREEN_BACK = 1000.0, 560.0
BEAM_UNDERSIDE, BEAM_BACK = 1341.0, 425.26820827052
# Spine: between the lift rails' front (Z 477.3) and the cradles, below the
# head's lowest sweep (Y 1254). 174 mm long to stay inside the build cube.
SPINE_HALF, SPINE_BOTTOM, SPINE_TOP = 20.0, 1058.0, 1232.0
SPINE_BACK, SPINE_FRONT = 490.0, 510.0
# Bolt axes (X, Y) through the spine, in the assembled positions.
HANGER_BOLTS = [(0, 1195), (0, 1220)]
LOWER_BOLT_Y = (1070, 1096)
KEEPER_BOLTS = [(0, 1142), (0, 1167)]
# Pockets 7 mm deep: an M5 x 25 with a washer goes 1 mm past its nut and stays
# inside the spine (the keeper shelf covers part of the upper front pocket).
M5_CLEAR, M5_NUT_AF, NUT_POCKET = 5.5, 8.4, 7.0
# The cradle and keeper bolt slots let them move this far up or down.
SLOT_TRAVEL = 2.0
# The cradle and keeper are drawn 20 mm high and moved down in run().
DROP = SCREEN_BOTTOM - 1020.0
# A1 mini build volume is 180 mm; keep room for a skirt.
BUILD_LIMIT = 175.0
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

    def sketch_point(self, sketch, axis, start, a, b):
        axes = [i for i in range(3) if i != axis]
        xyz = [0.0, 0.0, 0.0]
        xyz[axis], xyz[axes[0]], xyz[axes[1]] = start, a, b
        return sketch.modelToSketchSpace(self.local(xyz))

    def outline(self, sketch, axis, start, vertices):
        pts = [self.sketch_point(sketch, axis, start, a, b) for a, b in vertices]
        for p, q in zip(pts, pts[1:] + pts[:1]):
            line = sketch.sketchCurves.sketchLines.addByTwoPoints(p, q)
            line.isFixed = True

    def extrude(self, sketch, sign, start, end, name, operation, profiles):
        feature = self.comp.features.extrudeFeatures.addSimple(
            profiles,
            core.ValueInput.createByReal((end - start) / sign / 10),
            operation,
        )
        feature.name = name
        sketch.isVisible = False
        return feature

    def poly(self, axis, start, end, vertices, name, operation=NEW):
        sketch, sign = self.sketch(axis, start, name)
        self.outline(sketch, axis, start, vertices)
        assert sketch.profiles.count == 1, name
        return self.extrude(
            sketch, sign, start, end, name, operation, sketch.profiles.item(0)
        )

    def box(self, lo, hi, name):
        x, y, z = lo
        X, Y, Z = hi
        return self.poly(2, z, Z, [(x, y), (X, y), (X, Y), (x, Y)], name)

    def cut_all(self, sketch, sign, start, end, name):
        self.extrude(
            sketch, sign, start, end, name, CUT, collection(list(sketch.profiles))
        )

    def holes(self, axis, start, end, centers, diameter, name):
        sk, sign = self.sketch(axis, start, name)
        for a, b in centers:
            circle = sk.sketchCurves.sketchCircles.addByCenterRadius(
                self.sketch_point(sk, axis, start, a, b), diameter / 20
            )
            circle.isFixed = True
        self.cut_all(sk, sign, start, end, name)

    def slots(self, start, end, centers, width, name):
        # Through Z; each slot runs SLOT_TRAVEL up and down from its centre.
        sk, sign = self.sketch(2, start, name)
        r, t = width / 2, SLOT_TRAVEL
        for a, b in centers:

            def at(x, y):
                return self.sketch_point(sk, 2, start, x, y)

            lines, arcs = sk.sketchCurves.sketchLines, sk.sketchCurves.sketchArcs
            lines.addByTwoPoints(at(a - r, b - t), at(a - r, b + t)).isFixed = True
            arcs.addByThreePoints(
                at(a - r, b + t), at(a, b + t + r), at(a + r, b + t)
            ).isFixed = True
            lines.addByTwoPoints(at(a + r, b + t), at(a + r, b - t)).isFixed = True
            arcs.addByThreePoints(
                at(a + r, b - t), at(a, b - t - r), at(a - r, b - t)
            ).isFixed = True
        assert sk.profiles.count == len(centers), name
        self.cut_all(sk, sign, start, end, name)

    def nut_pockets(self, start, end, centers, name):
        # Hex pockets for M5 nuts through Z, corners up and down.
        sk, sign = self.sketch(2, start, name)
        r = M5_NUT_AF / math.sqrt(3)
        for a, b in centers:
            self.outline(
                sk,
                2,
                start,
                [
                    (
                        a + r * math.cos(math.radians(90 + 60 * k)),
                        b + r * math.sin(math.radians(90 + 60 * k)),
                    )
                    for k in range(6)
                ],
            )
        assert sk.profiles.count == len(centers), name
        self.cut_all(sk, sign, start, end, name)

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
    # only reaches forward below the head, where the spine starts.
    p = Part(root, "01_frame_adapter", [-28, 1180, BEAM_BACK])
    p.box((-28, 1333, BEAM_BACK), (28, 1341, 485), "beam_foot_8mm")
    p.box((-25, 1230, 434), (25, 1333.2, 440), "rear_drop_6mm")
    p.box((-25, 1224, 434), (25, 1232, 490), "dogleg_bridge_8mm")
    p.box((-25, 1180, 484), (25, 1232, 490), "spine_rear_plate_6mm")
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
    p.holes(2, 483.9, 490.1, HANGER_BOLTS, M5_CLEAR, "two_M5_spine_holes")
    return p


def make_spine(root):
    # Printed on its back face. The cradles and keeper bolt from the front into
    # nuts at the back; the adapter bolts from behind into nuts at the front.
    p = Part(root, "05_spine", [-SPINE_HALF, SPINE_BOTTOM, SPINE_BACK])
    p.box(
        (-SPINE_HALF, SPINE_BOTTOM, SPINE_BACK),
        (SPINE_HALF, SPINE_TOP, SPINE_FRONT),
        "spine_40x20",
    )
    p.join()
    front = [(x, y) for x in (-10, 10) for y in LOWER_BOLT_Y] + KEEPER_BOLTS
    p.holes(
        2,
        SPINE_BACK - 0.1,
        SPINE_FRONT + 0.1,
        front + HANGER_BOLTS,
        M5_CLEAR,
        "eight_M5_holes",
    )
    p.nut_pockets(SPINE_BACK - 0.1, SPINE_BACK + NUT_POCKET, front, "rear_M5_nuts")
    p.nut_pockets(
        SPINE_FRONT + 0.1, SPINE_FRONT - NUT_POCKET, HANGER_BOLTS, "front_M5_nuts"
    )
    return p


def make_lower(root, side):
    left = side < 0
    name = "02_lower_left" if left else "03_lower_right"
    p = Part(root, name, [-133.3 if left else 0.5, 1012, SPINE_FRONT])
    # The two halves bolt side by side on the spine front, 1 mm apart.
    low, high = SPINE_BOTTOM - DROP, SPINE_BOTTOM + 50 - DROP
    verts = [
        (0.5, low),
        (20, low),
        (95, 1018),
        (128.3, 1018),
        (128.3, 1080),
        (20, high),
        (0.5, high),
    ]
    if left:
        verts = [(-x, y) for x, y in verts][::-1]
    p.poly(2, 510, 518, verts, "spine_land_and_arm_8mm")

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
    bolts = [(side * 10, y - DROP) for y in LOWER_BOLT_Y]
    p.slots(509.9, 518.1, bolts, M5_CLEAR, "two_M5_spine_slots")
    p.slots(516, 518.1, bolts, 11, "M5_washer_recess_2mm")
    return p


def make_top(root):
    p = Part(root, "04_top_keeper", [-45, 1152, 510])
    p.box((-18, 1152, 510), (18, 1208.2, 518), "spine_land_8mm")
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
    bolts = [(x, y - DROP) for x, y in KEEPER_BOLTS]
    p.slots(509.9, 518.1, bolts, M5_CLEAR, "two_M5_spine_slots")
    p.slots(516, 518.1, bolts, 11, "M5_washer_recess_2mm")
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
    adapter = make_adapter(root)
    spine = make_spine(root)
    cradle = [make_lower(root, -1), make_lower(root, 1), make_top(root)]
    for p in cradle:
        t = p.occ.transform2
        v = t.translation
        v.y += DROP / 10
        t.translation = v
        p.occ.transform2 = t
        p.origin[1] += DROP
    parts = [adapter, *cradle, spine]
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
            if p.name.startswith(("01", "05")):
                # The long edges of the adapter and the spine.
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
        "display_vertical_offset_mm": DROP,
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
        assert max(size) <= BUILD_LIMIT, (p.name, size)
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
    assert em.execute(
        em.createFusionArchiveExportOptions(str(OUT / "display_mount_assembly.f3d"))
    )
    for path in OUT.glob("*.step"):
        path.write_text(
            "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
        )
    print(json.dumps(report, indent=2))

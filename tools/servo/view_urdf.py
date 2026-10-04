#!/usr/bin/env python3
"""URDF のゼロ姿勢と各関節の正方向を確認するビューア (ROS 不要)。
試す時点でROSが実装していないので、この方法で確認する。
最初に「全関節 0 rad」の姿勢を表示し、ウィンドウを閉じるごとに
「1 関節だけ +DEG°」の姿勢を順に表示する。各関節のフレームには
座標軸 (x=赤, y=緑, z=青) を描く。回転軸は z (青) で、
右手の親指を青軸の向きに合わせたときに指が巻く向きが + 方向。

関節とサーボの対応 (arm_controller.py の 6 スロット順):
  joint_?2 → joint0 (ID4–7)    joint_?3 → joint1 (ID8–11)
  joint_?4 → ID12              joint_?5 → ID13              joint_?6 → ID14

使い方 (WSLg などで GUI が出せる環境):
  uv run --with yourdfpy --with "pyglet<2" view_urdf.py --arm r
  uv run --with yourdfpy --with "pyglet<2" view_urdf.py --arm l --joints 4,5,6 --deg 45
"""

import argparse
import os

import trimesh
import yourdfpy

REPO = os.path.expanduser("~/aharobot/AhaRobot")
DESC = os.path.join(REPO, "upstream", "astra_description")
URDF_PATH = os.path.join(DESC, "urdf", "astra_description.urdf")
PKG = "package://astra_description/"

SLOT = {"2": "joint0 (ID4-7)", "3": "joint1 (ID8-11)", "4": "ID12", "5": "ID13", "6": "ID14"}


def handler(fname: str) -> str:
    if fname.startswith(PKG):
        return os.path.join(DESC, fname[len(PKG):])
    return fname


ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["r", "l"], default="r", help="r=右腕, l=左腕")
ap.add_argument("--joints", default="2,3,4,5,6", help="順に表示する関節番号")
ap.add_argument("--deg", type=float, default=30.0, help="正方向に回す角度")
args = ap.parse_args()

robot = yourdfpy.URDF.load(URDF_PATH, filename_handler=handler,
                           load_meshes=True, build_scene_graph=True)
names = [f"joint_{args.arm}{k}" for k in args.joints.split(",")]
for n in names:
    if n not in robot.joint_map:
        raise SystemExit(f"URDF に {n} がない")


def show(cfg: dict, caption: str) -> None:
    full = {n: 0.0 for n in robot.actuated_joint_names}
    full.update(cfg)
    robot.update_cfg(full)
    scene = robot.scene.copy()
    for n in names:
        T = robot.get_transform(robot.joint_map[n].child)
        scene.add_geometry(trimesh.creation.axis(
            origin_size=0.004, axis_radius=0.002, axis_length=0.06, transform=T))
    print(f"表示中: {caption}  (ウィンドウを閉じると次へ)")
    scene.show(caption=caption)


show({}, f"{args.arm}-arm: ALL ZERO (URDF 0 rad)")
for n in names:
    k = n[-1]
    show({n: __import__("math").radians(args.deg)},
         f"{n} = +{args.deg:.0f} deg  -> {SLOT.get(k, '?')}")
print("完了")

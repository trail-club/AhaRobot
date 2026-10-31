// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cmath>

#include "laundry/cloth.hh"
#include "laundry/fold_metrics.hh"
#include "laundry/pattern_io.hh"

using namespace aha_laundry;

namespace
{
constexpr double kTop = 0.75;

GarmentMesh Shirt()
{
  static const GarmentMesh mesh = MeshGarment(LoadPattern(TSHIRT_PATTERN), 0.035);
  return mesh;
}

Collider Table()
{
  Collider c;
  c.type = ShapeType::Box;
  c.label = "table";
  c.halfExtents = Vec3(0.6, 0.5, 0.0125);
  c.pose.translation() = Vec3(0, 0.36, kTop - 0.0125);
  c.previousPose = c.pose;
  c.friction = 0.6;
  return c;
}

ClothWorld TableWorld()
{
  ClothParameters p;
  p.bendCompliance = 1.0;
  ClothWorld world(p);
  world.Colliders().push_back(Table());
  return world;
}

void Simulate(ClothWorld &world, double seconds)
{
  for (int i = 0; i < std::lround(seconds / 0.005); ++i)
    world.Step(0.005);
}

int FlatShirt(ClothWorld &world)
{
  Pose pose = Pose::Identity();
  pose.translation() = Vec3(0, 0, kTop + 0.03);
  return world.AddGarment("shirt", Shirt(), pose);
}

GripperState Gripper()
{
  GripperState g;
  g.padA = Eigen::AlignedBox3d(Vec3(-0.012, -0.012, -0.02), Vec3(0.012, 0.012, -0.005));
  g.padB = Eigen::AlignedBox3d(Vec3(-0.012, -0.012, 0.005), Vec3(0.012, 0.012, 0.02));
  g.fingerB.translation() = Vec3(0, 0.2, 0);  // open
  return g;
}
}  // namespace

TEST(GarmentMesh, PanelsAreSewnAlongTheSeams)
{
  const auto mesh = Shirt();
  ASSERT_GT(mesh.rest2d.size(), 400u);
  size_t seam = 0;
  double mass = 0;
  for (size_t i = 0; i < mesh.rest2d.size(); ++i)
  {
    seam += mesh.particlePanel[i] == 2;
    mass += mesh.mass[i];
  }
  EXPECT_GT(seam, 40u);  // sides, under- and top-sleeves, shoulders
  EXPECT_NEAR(mass, 0.17, 1e-9);
  // A size M shirt laid flat covers about 0.45 m2 per panel.
  EXPECT_GT(mesh.frontArea, 0.35);
  EXPECT_LT(mesh.frontArea, 0.55);
  for (const auto &uv : mesh.renderUv)
  {
    EXPECT_GE(uv.x(), -1e-9);
    EXPECT_LE(uv.x(), 1 + 1e-9);
    EXPECT_GE(uv.y(), -1e-9);
    EXPECT_LE(uv.y(), 1 + 1e-9);
  }
  // Every particle belongs to a triangle; the front texture is the left half.
  std::vector<bool> used(mesh.rest2d.size(), false);
  for (const auto &t : mesh.triangles)
    for (auto v : t)
      used[v] = true;
  EXPECT_TRUE(std::all_of(used.begin(), used.end(), [](bool u) { return u; }));
}

TEST(Shapes, SignedDistancesAndNormals)
{
  Collider box = Table();
  Vec3 n;
  EXPECT_NEAR(ShapeDistance(box, Vec3(0, 0.36, kTop + 0.1), &n), 0.1, 1e-9);
  EXPECT_NEAR(n.z(), 1.0, 1e-9);
  EXPECT_LT(ShapeDistance(box, Vec3(0, 0.36, kTop - 0.005), &n), 0);
  Collider cylinder;
  cylinder.type = ShapeType::Cylinder;
  cylinder.radius = 0.2;
  cylinder.halfLength = 0.1;
  cylinder.pose = Pose::Identity();
  EXPECT_NEAR(ShapeDistance(cylinder, Vec3(0.5, 0, 0), &n), 0.3, 1e-9);
  EXPECT_NEAR(n.x(), 1.0, 1e-9);
  Vec3 bary;
  const Vec3 c = ClosestOnTriangle(Vec3(0.2, 0.2, 1), Vec3(0, 0, 0), Vec3(1, 0, 0),
                                   Vec3(0, 1, 0), &bary);
  EXPECT_NEAR((c - Vec3(0.2, 0.2, 0)).norm(), 0, 1e-12);
  EXPECT_NEAR(bary.sum(), 1, 1e-12);
}

TEST(ClothWorld, DroppedShirtLiesFlatWithSeparatedLayersAndSleeps)
{
  auto world = TableWorld();
  FlatShirt(world);
  Simulate(world, 2.0);
  const auto &g = world.Garments()[0];
  EXPECT_TRUE(g.asleep);
  double front = 0, back = 0;
  int nf = 0, nb = 0;
  for (size_t i = 0; i < g.x.size(); ++i)
  {
    EXPECT_GT(g.x[i].z(), kTop);
    EXPECT_LT(g.x[i].z(), kTop + 0.02);
    if (g.mesh.particlePanel[i] == 0) { front += g.x[i].z(); ++nf; }
    if (g.mesh.particlePanel[i] == 1) { back += g.x[i].z(); ++nb; }
  }
  // Self collision keeps the front panel on top of the back panel.
  EXPECT_GT(front / nf - back / nb, 0.5 * world.Parameters().thickness);
  const auto m = ComputeFoldMetrics(g, kTop, world.Parameters().thickness);
  EXPECT_NEAR(m.areaRatio, 1.0, 0.1);
  EXPECT_LT(m.foldedFraction, 0.05);
  EXPECT_NEAR(m.meanLayers, 2.0, 0.1);
}

TEST(ClothWorld, ClosingGripperPinchesAndLiftsCloth)
{
  auto world = TableWorld();
  FlatShirt(world);
  Simulate(world, 1.0);
  auto gripper = Gripper();
  const Vec3 at = world.Garments()[0].x[0];
  gripper.fingerA.translation() = at;
  gripper.previousFingerA = gripper.fingerA;
  gripper.fingerB.translation() = at + Vec3(0, 0.2, 0);
  world.Grippers().push_back(gripper);
  EXPECT_EQ(world.UpdateGrasps(0.012, 0.02, 0.006), 0);  // open: nothing held
  world.Grippers()[0].fingerB = world.Grippers()[0].fingerA;
  EXPECT_GT(world.UpdateGrasps(0.012, 0.02, 0.006), 0);
  for (int i = 1; i <= 200; ++i)
  {
    world.Grippers()[0].fingerA.translation() = at + Vec3(0, 0, 0.3 * i / 200.0);
    world.Grippers()[0].fingerB = world.Grippers()[0].fingerA;
    world.Step(0.005);
  }
  double top = 0;
  for (const auto &p : world.Garments()[0].x)
    top = std::max(top, p.z());
  EXPECT_GT(top, kTop + 0.25);
  EXPECT_FALSE(world.Garments()[0].asleep);
  world.Grippers()[0].fingerB.translation() += Vec3(0, 0.2, 0);
  world.UpdateGrasps(0.012, 0.02, 0.006);
  EXPECT_TRUE(world.Attachments().empty());
}

TEST(ClothWorld, CrumpledPileIsCompactAndBunched)
{
  ClothParameters p;
  p.bendCompliance = 1.0;
  const auto x = CrumpleGarment(Shirt(), p, 3, 0.17);
  auto world = TableWorld();
  const int g = world.AddGarment("shirt", Shirt(), Pose::Identity());
  auto placed = x;
  for (auto &q : placed)
    q += Vec3(0, 0.36, kTop + 0.005);
  world.SetPose(g, placed);
  Simulate(world, 1.0);
  const auto m = ComputeFoldMetrics(world.Garments()[0], kTop, p.thickness);
  EXPECT_LT(m.areaRatio, 0.7);
  EXPECT_GT(m.heightP95 - m.expectedHeight, 0.05);
}

TEST(FoldMetrics, FoldingInHalfDoublesTheLayers)
{
  auto world = TableWorld();
  FlatShirt(world);
  Simulate(world, 1.0);
  auto &g = world.Garments()[0];
  const double thickness = world.Parameters().thickness;
  // Mirror the lower half over the fold line y = 0.36: an ideal half fold.
  std::vector<Vec3> folded = g.x;
  for (auto &p : folded)
    if (p.y() < 0.36)
      p = Vec3(p.x(), 0.72 - p.y(), kTop + 4 * thickness - (p.z() - kTop));
  world.SetPose(0, folded);
  const auto m = ComputeFoldMetrics(world.Garments()[0], kTop, thickness);
  EXPECT_GT(m.foldedFraction, 0.6);
  EXPECT_LT(m.areaRatio, 0.75);
  EXPECT_LT(m.widthRatio, 0.75);
  EXPECT_NEAR(m.meanLayers, 3.5, 0.6);
  auto other = m;
  EXPECT_NEAR(FootprintOverlap(m, other), 1.0, 1e-12);
}

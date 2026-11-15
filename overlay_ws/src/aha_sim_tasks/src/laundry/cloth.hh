// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

// Position-based (XPBD) simulation of sewn garments. The solver has no
// Gazebo dependency: the laundry system feeds it rigid colliders and gripper
// poses and reads particle positions back for rendering and scoring.

#ifndef AHA_SIM_TASKS_LAUNDRY_CLOTH_HH_
#define AHA_SIM_TASKS_LAUNDRY_CLOTH_HH_

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Geometry>

namespace aha_laundry
{
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Pose = Eigen::Isometry3d;

// A flat sewing-pattern panel. Outline vertices are counter-clockwise and
// each edge i (vertex i to i+1) is either sewn to the other panel or open.
struct PanelPattern
{
  std::vector<Vec2> outline;
  std::vector<bool> seam;  // per outline edge
  // Texture coordinates: uv = uvOffset + uvScale .* (x, y).
  Vec2 uvOffset{0, 0};
  Vec2 uvScale{1, 1};
};

struct GarmentPattern
{
  std::string name;
  PanelPattern front;
  PanelPattern back;
  double mass = 0.16;  // kg
};

// Triangulated, sewn garment in its flat rest configuration.
struct GarmentMesh
{
  std::vector<Vec2> rest2d;  // pattern coordinates per particle
  std::vector<double> mass;
  std::vector<std::array<uint32_t, 3>> triangles;
  std::vector<uint8_t> trianglePanel;  // 0 front, 1 back
  std::vector<uint8_t> particlePanel;  // 0 front, 1 back, 2 seam
  std::vector<std::array<uint32_t, 2>> edges;
  std::vector<double> edgeRest;
  std::vector<std::array<uint32_t, 2>> bends;  // opposite vertices of a hinge
  std::vector<double> bendRest;
  // Render vertices keep one texture coordinate per panel.
  std::vector<uint32_t> renderParticle;
  std::vector<Vec2> renderUv;
  std::vector<std::array<uint32_t, 3>> renderTriangles;
  double frontArea = 0;  // flat area of one panel, the footprint of a spread shirt
  Vec2 extent{0, 0};  // flat pattern size (width, length)
};

// Builds a two-panel mesh, merging particles along the seams.
GarmentMesh MeshGarment(const GarmentPattern &pattern, double spacing);

enum class ShapeType { Box, Sphere, Capsule, Cylinder, Plane };

// Rigid obstacle; shapes are centred at the pose origin, capsules and
// cylinders along local z, planes face local +z.
struct Collider
{
  ShapeType type = ShapeType::Box;
  Vec3 halfExtents{0, 0, 0};
  double radius = 0;
  double halfLength = 0;
  double friction = 0.5;
  Pose pose = Pose::Identity();
  Pose previousPose = Pose::Identity();
  int body = -1;  // dynamic rigid body receiving cloth reactions, or -1
  int gripper = -1;  // gripper this collider belongs to, or -1
  bool movable = false;  // may move; wakes sleeping cloth when it does
  std::string label;
  // Filled by the solver.
  Eigen::AlignedBox3d bounds;
  bool moved = false;
};

struct ClothParameters
{
  double thickness = 0.004;  // separation between touching layers
  double contactOffset = 0.0025;  // particle radius against rigid shapes
  double stretchCompliance = 1e-8;
  double bendCompliance = 2e-4;
  double damping = 0.6;  // 1/s velocity damping
  double clothFriction = 0.45;
  double maxSpeed = 3.0;
  int substeps = 4;
  bool selfCollision = true;
  double contactRelaxation = 0.6;  // fraction of a cloth-cloth overlap removed per substep
  int maxContacts = 6;  // cloth contacts kept per particle
  double contactSkin = 0.01;  // search margin; pairs are reused while motion stays below half
  double contactDamping = 0.05;  // velocity removed per substep from pressed layers
  int exclusionRings = 2;  // mesh neighbourhood ignored by self collision
  double sleepSpeed = 0.04;
  double sleepDelay = 0.6;
  Vec3 gravity{0, 0, -9.81};
};

// Finger pads of a parallel gripper, expressed in each finger link frame.
// The closing direction of finger A points from A towards B.
struct GripperState
{
  Pose fingerA = Pose::Identity();
  Pose fingerB = Pose::Identity();
  Pose previousFingerA = Pose::Identity();
  Eigen::AlignedBox3d padA;  // pad volume in finger A frame
  Eigen::AlignedBox3d padB;
  double gap = 1.0;  // distance between inner pad faces
  bool closed = false;
};

struct Attachment
{
  int garment;
  uint32_t particle;
  int gripper;
  Vec3 local;  // particle in finger A frame
};

struct Garment
{
  std::string id;
  GarmentMesh mesh;
  std::vector<Vec3> x, v, previous;
  std::vector<Vec3> stepStart;  // positions when the current step began
  std::vector<double> invMass;
  std::vector<std::vector<uint32_t>> nearby;  // topological neighbours, sorted
  bool asleep = false;
  double calmTime = 0;
  double windowTime = 0;  // centre displacement is checked once per sleep delay
  Vec3 windowCentre = Vec3::Constant(1e9);
  Eigen::AlignedBox3d bounds;
  // Contacts in the most recent step, per collider label.
  std::unordered_map<std::string, int> contacts;
  int clothContacts = 0;
  // Displacement over the last step divided by its duration: oscillations
  // inside a step cancel, unlike the substep velocities.
  double maxSpeed = 0;
  double meanSpeed = 0;
  Vec3 bodyImpulse = Vec3::Zero();  // pushed onto dynamic bodies in the last step
};

class ClothWorld
{
 public:
  explicit ClothWorld(ClothParameters parameters = {});

  int AddGarment(const std::string &id, const GarmentMesh &mesh, const Pose &pose);
  std::vector<Garment> &Garments() { return this->garments; }
  const std::vector<Garment> &Garments() const { return this->garments; }
  std::vector<Collider> &Colliders() { return this->colliders; }
  std::vector<GripperState> &Grippers() { return this->grippers; }
  const std::vector<Attachment> &Attachments() const { return this->attachments; }
  const ClothParameters &Parameters() const { return this->parameters; }
  ClothParameters &MutableParameters() { return this->parameters; }

  // Advance by dt. Collider and gripper poses must already hold their values
  // at the end of the step; previous poses are interpolated within substeps.
  void Step(double dt);
  // Grippers that just closed pinch the cloth between their pads; opened
  // grippers release it. Returns the number of particles newly attached.
  int UpdateGrasps(double closeGap, double openGap, double margin);
  void ReleaseGripper(int gripper);

  // Rigid-body reaction impulses accumulated since the last call, per body,
  // as {force impulse, torque impulse about the world origin}.
  std::unordered_map<int, std::pair<Vec3, Vec3>> TakeBodyImpulses();
  void Wake(int garment);
  void SetPose(int garment, const std::vector<Vec3> &positions);

 private:
  struct ClothContact
  {
    int garment;
    uint32_t particle;
    int triangleGarment;
    uint32_t triangle;
    double side;
  };

  void Predict(Garment &garment, double h);
  void SolveDistances(Garment &garment, double h);
  void SolveShapes(Garment &garment, double h, double fraction);
  void SolveClothContacts(double h);
  void DampClothContacts();
  void FindClothContacts(double dt);
  void ApplyAttachments(double fraction);
  void UpdateSleep(double dt);
  bool Attached(int garment, uint32_t particle) const;
  std::vector<bool> AwakeMask() const;

  ClothParameters parameters;
  std::vector<Garment> garments;
  std::vector<Collider> colliders;
  std::vector<GripperState> grippers;
  std::vector<Attachment> attachments;
  std::vector<ClothContact> clothContacts;
  std::vector<std::vector<uint8_t>> attachedMask;
  std::unordered_map<int, std::pair<Vec3, Vec3>> bodyImpulses;
  // Reused spatial-hash storage for the cloth contact search.
  struct HashSpan
  {
    int garment;
    uint32_t triangle;
    int64_t lower[3], upper[3];
  };
  std::vector<HashSpan> hashSpans;
  std::vector<uint32_t> hashStart, hashFill, hashItems;
  std::vector<Vec3> hashPoints;
  std::vector<std::pair<double, ClothContact>> hashBest;
  std::vector<bool> searchAwake;
  std::vector<std::vector<Vec3>> searchPositions;
};

// Signed distance and outward normal from a collider surface, world frame.
double ShapeDistance(const Collider &collider, const Vec3 &point, Vec3 *normal);

// Boxes approximating a cylindrical wall about the local z axis.
std::vector<Collider> Tube(const Pose &pose, double radius, double height, int sides,
                           const std::string &label);

// A garment hung from a random point and lowered in a spiral into a tube of
// the given radius, as when laundry is dropped. Positions are relative to the
// pile's centre on the floor.
std::vector<Vec3> CrumpleGarment(const GarmentMesh &mesh, const ClothParameters &parameters,
                                 uint32_t seed, double radius);

// Closest point on triangle abc to p, returning barycentric weights.
Vec3 ClosestOnTriangle(const Vec3 &p, const Vec3 &a, const Vec3 &b, const Vec3 &c,
                       Vec3 *barycentric);
}  // namespace aha_laundry

#endif

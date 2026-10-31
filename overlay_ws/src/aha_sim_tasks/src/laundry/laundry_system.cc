// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

// Doing Laundry challenge: simulates the T-shirts as cloth, renders them for
// the robot cameras, and publishes the garment states used for scoring.
// The evaluation never moves the cloth or the furniture; only the arena door
// is opened, as the referee does at the start signal.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include <gz/common/Console.hh>
#include <gz/common/Mesh.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/marker.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/rendering/RenderingIface.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/Geometry.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/JointVelocityCmd.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/Static.hh>
#include <gz/sim/rendering/Events.hh>
#include <gz/transport/Node.hh>
#include <nlohmann/json.hpp>
#include <sdf/Box.hh>
#include <sdf/Capsule.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Geometry.hh>
#include <sdf/Mesh.hh>
#include <sdf/Sphere.hh>

#include "cloth.hh"
#include "cloth_renderer.hh"
#include "fold_metrics.hh"
#include "pattern_io.hh"

namespace aha_laundry
{
namespace
{
using nlohmann::json;
using gz::sim::Entity;
using gz::sim::EntityComponentManager;
using gz::sim::kNullEntity;

Pose ToPose(const gz::math::Pose3d &p)
{
  Pose pose = Pose::Identity();
  pose.translation() = Vec3(p.Pos().X(), p.Pos().Y(), p.Pos().Z());
  pose.linear() = Eigen::Quaterniond(p.Rot().W(), p.Rot().X(), p.Rot().Y(), p.Rot().Z())
                      .toRotationMatrix();
  return pose;
}

Vec3 ToVec(const json &v)
{
  return Vec3(v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>());
}

json Stamp(std::chrono::steady_clock::duration time)
{
  const auto sec = std::chrono::duration_cast<std::chrono::seconds>(time);
  const auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(time - sec);
  return {{"sec", sec.count()}, {"nanosec", nsec.count()}};
}

double Seconds(std::chrono::steady_clock::duration time)
{
  return std::chrono::duration<double>(time).count();
}

struct GarmentConfig
{
  std::string id;
  std::string pattern;
  std::string albedo;
  std::string normal;
  std::string origin;
  std::array<double, 3> color{0.5, 0.5, 0.5};
};

struct Shape
{
  Collider collider;
  Vec3 center = Vec3::Zero();  // offset of a mesh box from its collision frame
};

struct ColliderSource
{
  Entity link = kNullEntity;
  Pose local = Pose::Identity();
  size_t index = 0;  // in ClothWorld::Colliders()
};
}  // namespace

class LaundrySystem : public gz::sim::System,
                      public gz::sim::ISystemConfigure,
                      public gz::sim::ISystemPreUpdate,
                      public gz::sim::ISystemPostUpdate
{
 public:
  ~LaundrySystem() override
  {
    this->preRender.reset();
    this->teardown.reset();
  }

  void Configure(const Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 EntityComponentManager &, gz::sim::EventManager &events) override
  {
    const auto path = sdf->Get<std::string>("scene_config", "").first;
    try
    {
      std::ifstream stream(path);
      if (!stream)
        throw std::runtime_error("cannot read " + path);
      this->scene = json::parse(stream);
      this->config = this->scene.at("laundry");
      this->LoadConfig();
    }
    catch (const std::exception &error)
    {
      gzerr << "[aha_laundry] Invalid scene configuration " << path << ": " << error.what()
            << '\n';
      return;
    }
    this->configured = true;
    this->preRender = events.Connect<gz::sim::events::PreRender>(
        [this]() { this->renderer.Update(gz::rendering::sceneFromFirstRenderEngine()); });
    this->teardown = events.Connect<gz::sim::events::RenderTeardown>(
        [this]() { this->renderer.Destroy(); });
    this->node.Subscribe("/evaluation/start", &LaundrySystem::OnStart, this);
  }

  void PreUpdate(const gz::sim::UpdateInfo &info, EntityComponentManager &ecm) override
  {
    if (!this->configured || info.paused)
      return;
    if (!this->initialized)
    {
      if (!this->Initialize(ecm))
        return;
      this->lastCloth = info.simTime;
    }
    if (!this->robotReady)
      this->AttachRobot(ecm);
    this->ControlArenaDoor(info, ecm);
    const double elapsed = Seconds(info.simTime - this->lastCloth);
    if (info.simTime < this->lastCloth)
      this->lastCloth = info.simTime;
    if (elapsed + 1e-9 >= this->clothStep)
    {
      const auto clothStarted = std::chrono::steady_clock::now();
      this->UpdateColliders(ecm);
      this->world.UpdateGrasps(this->closeGap, this->openGap, this->pinchMargin);
      this->world.Step(elapsed);
      this->clothWall += std::chrono::steady_clock::now() - clothStarted;
      if (info.simTime - this->lastReport >= std::chrono::seconds(10))
      {
        int awake = 0;
        std::ostringstream which;
        for (const auto &g : this->world.Garments())
          if (!g.asleep)
          {
            ++awake;
            which << ' ' << g.id << '(' << g.meanSpeed << '/' << g.maxSpeed << ')';
          }
        gzdbg << "[aha_laundry] cloth took " << std::chrono::duration<double>(this->clothWall).count()
              << " s wall for " << Seconds(info.simTime - this->lastReport) << " s simulated; "
              << awake << " garments awake" << which.str() << '\n';
        this->clothWall = {};
        this->lastReport = info.simTime;
      }
      this->lastCloth = info.simTime;
      this->clothChanged = true;
      this->basketWrench = {Vec3::Zero(), Vec3::Zero()};
      for (const auto &[body, impulse] : this->world.TakeBodyImpulses())
        this->basketWrench = {impulse.first / elapsed, impulse.second / elapsed};
      if (!this->basketWrench.first.allFinite() || !this->basketWrench.second.allFinite())
      {
        gzwarn << "[aha_laundry] Ignoring a non-finite cloth reaction on the basket\n";
        this->basketWrench = {Vec3::Zero(), Vec3::Zero()};
      }
      else if (this->basketWrench.first.norm() > this->maxBasketForce)
      {
        static int reports = 0;
        if (reports++ < 3)
          gzwarn << "[aha_laundry] Limiting cloth reaction on the basket: "
                 << this->basketWrench.first.norm() << " N\n";
      }
      this->TrackFloorContacts();
    }
    if (this->basketLink != kNullEntity)
    {
      // Cloth resting in or pushing on the basket loads it like real laundry.
      const auto pose = gz::sim::worldPose(this->basketLink, ecm);
      const Vec3 origin(pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z());
      Vec3 force = this->basketWrench.first;
      Vec3 torque = this->basketWrench.second - origin.cross(force);
      // A pile of shirts weighs about 1.5 kg; larger reactions come from
      // resolving initial overlaps and must not launch the basket.
      if (force.norm() > this->maxBasketForce)
        force *= this->maxBasketForce / force.norm();
      if (torque.norm() > this->maxBasketTorque)
        torque *= this->maxBasketTorque / torque.norm();
      gz::sim::Link(this->basketLink).AddWorldWrench(
          ecm, gz::math::Vector3d(force.x(), force.y(), force.z()),
          gz::math::Vector3d(torque.x(), torque.y(), torque.z()));
    }
  }

  void PostUpdate(const gz::sim::UpdateInfo &info, const EntityComponentManager &ecm) override
  {
    if (!this->initialized || info.paused)
      return;
    if (this->clothChanged && info.simTime - this->lastRender >= std::chrono::milliseconds(33))
    {
      this->PublishRender(false);
      this->lastRender = info.simTime;
      this->clothChanged = false;
    }
    if (info.simTime - this->lastMarkers >= std::chrono::milliseconds(250) ||
        info.simTime < this->lastMarkers)
    {
      this->PublishMarkers();
      this->lastMarkers = info.simTime;
    }
    if (info.simTime < this->lastState)
      this->lastState = info.simTime;
    if (info.simTime - this->lastState < std::chrono::milliseconds(50))
      return;
    this->lastState = info.simTime;
    this->PublishState(info, ecm);
  }

 private:
  void LoadConfig()
  {
    const auto &cloth = this->config.at("cloth");
    ClothParameters p;
    p.thickness = cloth.value("thickness", p.thickness);
    p.contactOffset = cloth.value("contact_offset", p.contactOffset);
    p.stretchCompliance = cloth.value("stretch_compliance", p.stretchCompliance);
    p.bendCompliance = cloth.value("bend_compliance", p.bendCompliance);
    p.damping = cloth.value("damping", p.damping);
    p.clothFriction = cloth.value("cloth_friction", p.clothFriction);
    p.substeps = cloth.value("substeps", p.substeps);
    p.sleepSpeed = cloth.value("sleep_speed", p.sleepSpeed);
    p.sleepDelay = cloth.value("sleep_delay", p.sleepDelay);
    p.contactRelaxation = cloth.value("contact_relaxation", p.contactRelaxation);
    this->parameters = p;
    this->world = ClothWorld(p);
    this->spacing = cloth.value("spacing", 0.035);
    this->clothStep = cloth.value("step", 0.005);
    this->closeGap = cloth.value("grasp_close_gap", 0.013);
    this->openGap = cloth.value("grasp_open_gap", 0.022);
    this->pinchMargin = cloth.value("pinch_margin", 0.006);
    this->friction = cloth.value("friction", json::object());
    this->seed = this->scene.value("seed", 0);
    for (const auto &entry : this->config.at("garments"))
      this->garments.push_back({entry.at("id"), entry.at("pattern"), entry.value("albedo", ""),
                                entry.value("normal", ""), entry.at("origin"),
                                entry.value("color", std::array<double, 3>{0.5, 0.5, 0.5})});
    this->robotName = this->scene.at("entities").at("robot_model");
  }

  double FrictionFor(const std::string &label) const
  {
    for (const auto &[key, value] : this->friction.items())
      if (label.rfind(key, 0) == 0)
        return value.get<double>();
    return 0.5;
  }

  Entity ModelOf(const std::string &name, EntityComponentManager &ecm) const
  {
    return ecm.EntityByComponents(gz::sim::components::Model(),
                                  gz::sim::components::Name(name));
  }

  // Converts one Gazebo collision into a cloth collider.
  std::optional<Shape> MakeCollider(const sdf::Geometry &geometry) const
  {
    Shape shape;
    Collider &c = shape.collider;
    switch (geometry.Type())
    {
      case sdf::GeometryType::BOX:
      {
        const auto size = geometry.BoxShape()->Size();
        c.type = ShapeType::Box;
        c.halfExtents = Vec3(size.X(), size.Y(), size.Z()) / 2;
        return shape;
      }
      case sdf::GeometryType::SPHERE:
        c.type = ShapeType::Sphere;
        c.radius = geometry.SphereShape()->Radius();
        return shape;
      case sdf::GeometryType::CYLINDER:
        c.type = ShapeType::Cylinder;
        c.radius = geometry.CylinderShape()->Radius();
        c.halfLength = geometry.CylinderShape()->Length() / 2;
        return shape;
      case sdf::GeometryType::CAPSULE:
        c.type = ShapeType::Capsule;
        c.radius = geometry.CapsuleShape()->Radius();
        c.halfLength = geometry.CapsuleShape()->Length() / 2;
        return shape;
      case sdf::GeometryType::PLANE:
        c.type = ShapeType::Plane;
        return shape;
      case sdf::GeometryType::MESH:
      {
        // Robot links: an oriented box around the collision mesh.
        const auto *mesh = gz::sim::loadMesh(*geometry.MeshShape());
        if (mesh == nullptr)
          return std::nullopt;
        const auto scale = geometry.MeshShape()->Scale();
        const auto lower = mesh->Min(), upper = mesh->Max();
        c.type = ShapeType::Box;
        c.halfExtents = Vec3((upper.X() - lower.X()) * scale.X(),
                             (upper.Y() - lower.Y()) * scale.Y(),
                             (upper.Z() - lower.Z()) * scale.Z()) / 2;
        shape.center = Vec3((upper.X() + lower.X()) * scale.X(),
                            (upper.Y() + lower.Y()) * scale.Y(),
                            (upper.Z() + lower.Z()) * scale.Z()) / 2;
        return shape;
      }
      default:
        return std::nullopt;
    }
  }

  void AddColliders(Entity model, const std::string &label, bool robot,
                    EntityComponentManager &ecm, std::vector<Collider> &out,
                    std::vector<ColliderSource> &sources)
  {
    const bool isStatic = ecm.Component<gz::sim::components::Static>(model) != nullptr &&
                          ecm.Component<gz::sim::components::Static>(model)->Data();
    for (const auto link : ecm.ChildrenByComponents(model, gz::sim::components::Link()))
    {
      const auto linkName = ecm.Component<gz::sim::components::Name>(link)->Data();
      const std::string colliderLabel = label + ":" + linkName;
      if (robot && this->padOverrides.count(linkName))
      {
        // Finger pads replace the coarse box around the whole finger mesh.
        for (const auto &pad : this->padOverrides.at(linkName))
        {
          Collider c;
          c.type = ShapeType::Box;
          c.halfExtents = pad.sizes() / 2;
          c.label = colliderLabel;
          c.movable = true;
          c.friction = this->FrictionFor(colliderLabel);
          c.gripper = this->fingerGripper.at(linkName);
          ColliderSource source;
          source.link = link;
          source.local.translation() = pad.center();
          source.index = out.size();
          c.pose = ToPose(gz::sim::worldPose(link, ecm)) * source.local;
          c.previousPose = c.pose;
          sources.push_back(source);
          out.push_back(c);
        }
        continue;
      }
      for (const auto collision :
           ecm.ChildrenByComponents(link, gz::sim::components::Collision()))
      {
        const auto *geometry = ecm.Component<gz::sim::components::Geometry>(collision);
        if (geometry == nullptr)
          continue;
        auto shape = this->MakeCollider(geometry->Data());
        if (!shape)
          continue;
        Collider *collider = &shape->collider;
        if (robot && geometry->Data().Type() != sdf::GeometryType::MESH &&
            geometry->Data().Type() != sdf::GeometryType::BOX)
          continue;  // wheels and casters never reach the cloth
        const auto *pose = ecm.Component<gz::sim::components::Pose>(collision);
        Pose local = pose ? ToPose(pose->Data()) : Pose::Identity();
        local.translation() += local.linear() * shape->center;
        collider->label = colliderLabel;
        collider->movable = !isStatic;
        collider->friction = this->FrictionFor(colliderLabel);
        if (link == this->basketLink)
          collider->body = 0;
        ColliderSource source{link, local, out.size()};
        collider->pose = ToPose(gz::sim::worldPose(link, ecm)) * local;
        collider->previousPose = collider->pose;
        sources.push_back(source);
        out.push_back(*collider);
      }
    }
  }

  bool Initialize(EntityComponentManager &ecm)
  {
    const auto &regions = this->config.at("regions");
    const auto washer = this->ModelOf(regions.at("washer").at("model"), ecm);
    const auto basket = this->ModelOf(regions.at("basket").at("model"), ecm);
    const auto table = this->ModelOf(regions.at("table").at("model"), ecm);
    if (washer == kNullEntity || basket == kNullEntity || table == kNullEntity)
      return false;
    this->washerBody = gz::sim::Model(washer).LinkByName(ecm, regions.at("washer").at("link"));
    this->basketLink = gz::sim::Model(basket).LinkByName(ecm, regions.at("basket").at("link"));
    this->tableLink = gz::sim::Model(table).LinkByName(ecm, regions.at("table").at("link"));
    if (this->washerBody == kNullEntity || this->basketLink == kNullEntity ||
        this->tableLink == kNullEntity)
      return false;
    const auto doorModel = this->ModelOf(regions.at("washer").value("door_model", ""), ecm);
    if (doorModel == kNullEntity)
      return false;
    this->washerDoor = gz::sim::Model(doorModel).JointByName(ecm, regions.at("washer").at("door_joint"));
    const auto doorLink = gz::sim::Model(doorModel).LinkByName(
        ecm, regions.at("washer").value("door_link", "door"));
    if (doorLink != kNullEntity)
    {
      // The door link origin is on the hinge; its axis points along -z.
      this->doorHinge = ToPose(gz::sim::worldPose(doorLink, ecm));
    }
    if (this->config.contains("arena_door"))
    {
      const auto door = this->ModelOf(this->config["arena_door"].at("model"), ecm);
      if (door != kNullEntity)
        this->arenaDoor = gz::sim::Model(door).JointByName(ecm, this->config["arena_door"].at("joint"));
    }
    for (const auto joint : {this->washerDoor, this->arenaDoor})
      if (joint != kNullEntity && !ecm.Component<gz::sim::components::JointPosition>(joint))
        ecm.CreateComponent(joint, gz::sim::components::JointPosition());
    // Every non-robot model with a collision becomes an obstacle for the cloth.
    std::vector<Collider> colliders;
    const std::set<std::string> ignored = this->config.value("ignore_models", std::set<std::string>{});
    ecm.Each<gz::sim::components::Model, gz::sim::components::Name,
             gz::sim::components::ParentEntity>(
        [&](const Entity &model, const gz::sim::components::Model *,
            const gz::sim::components::Name *name, const gz::sim::components::ParentEntity *)
        {
          if (name->Data() == this->robotName || ignored.count(name->Data()))
            return true;
          if (gz::sim::topLevelModel(model, ecm) != model)
            return true;
          this->AddColliders(model, name->Data(), false, ecm, colliders, this->sources);
          return true;
        });
    for (auto &c : colliders)
      if (c.label.rfind("ground", 0) == 0 && c.type == ShapeType::Plane)
        c.label = "floor";
    this->world.Colliders() = colliders;
    this->ResolveRegions(ecm);
    try
    {
      this->CreateGarments();
    }
    catch (const std::exception &error)
    {
      gzerr << "[aha_laundry] Cannot create garments: " << error.what() << '\n';
      this->configured = false;
      return false;
    }
    this->initialized = true;
    this->PublishRender(true);
    gzmsg << "[aha_laundry] " << this->world.Garments().size() << " garments, "
          << this->world.Colliders().size() << " colliders ready\n";
    return true;
  }

  void ResolveRegions(EntityComponentManager &ecm)
  {
    const auto &regions = this->config.at("regions");
    const auto &washer = regions.at("washer");
    const Pose body = ToPose(gz::sim::worldPose(this->washerBody, ecm));
    this->drumCenter = body * ToVec(washer.at("drum_center"));
    this->drumAxis = (body.linear() * ToVec(washer.at("drum_axis"))).normalized();
    this->drumRadius = washer.at("drum_radius");
    this->drumDepth = washer.at("drum_depth");
    const auto &basket = regions.at("basket");
    this->basketInteriorCenter = ToVec(basket.at("interior_center"));
    this->basketInteriorHalf = ToVec(basket.at("interior_half"));
    const auto &table = regions.at("table");
    const Pose tablePose = ToPose(gz::sim::worldPose(this->tableLink, ecm));
    this->tableTop = tablePose * ToVec(table.at("top_center"));
    this->tableYaw = std::atan2(tablePose.linear()(1, 0), tablePose.linear()(0, 0));
    this->tableHalf = Vec2(table.at("top_half").at(0), table.at("top_half").at(1));
    this->basketPose = ToPose(gz::sim::worldPose(this->basketLink, ecm));
  }

  bool InDrum(const Vec3 &p) const
  {
    const Vec3 d = p - this->drumCenter;
    const double along = d.dot(this->drumAxis);
    return std::abs(along) <= this->drumDepth / 2 + 0.02 &&
           (d - along * this->drumAxis).norm() <= this->drumRadius + 0.01;
  }

  bool InBasket(const Vec3 &p) const
  {
    const Vec3 local = this->basketPose.inverse() * p - this->basketInteriorCenter;
    return (local.cwiseAbs() - this->basketInteriorHalf - Vec3::Constant(0.01)).maxCoeff() <= 0;
  }

  bool OnTable(const Vec3 &p) const
  {
    const Vec3 d = p - this->tableTop;
    const double c = std::cos(this->tableYaw), s = std::sin(this->tableYaw);
    const double u = c * d.x() + s * d.y(), v = -s * d.x() + c * d.y();
    return std::abs(u) <= this->tableHalf.x() + 0.005 && std::abs(v) <= this->tableHalf.y() + 0.005 &&
           d.z() >= -0.005 && d.z() <= 0.4;
  }

  // ---------------------------------------------------------------- settling
  std::vector<Vec3> CrumplePile(const GarmentMesh &mesh, uint32_t seed, double radius) const
  {
    return CrumpleGarment(mesh, this->parameters, seed, radius);
  }

  static double PileHeight(const std::vector<Vec3> &x)
  {
    double top = 0;
    for (const auto &q : x)
      top = std::max(top, q.z());
    return top;
  }

  void SettleGroup(ClothWorld &w, double seconds, double minimum) const
  {
    const double dt = 0.005;
    for (int i = 0; i < std::lround(seconds / dt); ++i)
    {
      w.Step(dt);
      if (i * dt >= minimum &&
          std::all_of(w.Garments().begin(), w.Garments().end(),
                      [](const Garment &g) { return g.asleep; }))
        break;
    }
  }

  std::string CachePath(const std::vector<GarmentMesh> &meshes) const
  {
    json key = {{"version", 7}, {"seed", this->seed}, {"cloth", this->config.at("cloth")},
                {"garments", this->config.at("garments")},
                {"placement", this->config.value("placement", json::object())},
                {"drum", {this->drumCenter.x(), this->drumCenter.y(), this->drumCenter.z()}},
                {"basket", {this->basketPose.translation().x(), this->basketPose.translation().y()}},
                {"table", {this->tableTop.x(), this->tableTop.y(), this->tableTop.z()}}};
    for (auto &g : key["garments"])
    {
      // Paths differ between installs; the pattern file name identifies the size.
      g["pattern"] = std::filesystem::path(g.value("pattern", "")).filename().string();
      g.erase("albedo");
      g.erase("normal");
    }
    size_t particles = 0;
    for (const auto &m : meshes)
      particles += m.rest2d.size();
    key["particles"] = particles;
    std::string directory;
    if (const char *env = std::getenv("AHA_LAUNDRY_CACHE_DIR"))
      directory = env;
    else if (const char *home = std::getenv("HOME"))
      directory = std::string(home) + "/.cache/aha_sim_tasks/laundry";
    else
      return "";
    return directory + "/" + std::to_string(std::hash<std::string>{}(key.dump())) + ".bin";
  }

  void CreateGarments()
  {
    std::vector<GarmentMesh> meshes;
    std::map<std::string, GarmentMesh> byPattern;
    for (const auto &g : this->garments)
    {
      if (!byPattern.count(g.pattern))
        byPattern[g.pattern] = MeshGarment(LoadPattern(g.pattern), this->spacing);
      meshes.push_back(byPattern[g.pattern]);
    }
    for (size_t i = 0; i < meshes.size(); ++i)
    {
      Pose far = Pose::Identity();
      far.translation() = Vec3(100 + 2.0 * i, 100, 1);
      this->world.AddGarment(this->garments[i].id, meshes[i], far);
    }
    const auto cache = this->CachePath(meshes);
    std::vector<std::vector<Vec3>> settled;
    if (!cache.empty() && this->LoadCache(cache, meshes, settled))
    {
      gzmsg << "[aha_laundry] Loaded settled laundry from " << cache << '\n';
    }
    else
    {
      const auto started = std::chrono::steady_clock::now();
      settled = this->Settle(meshes);
      gzmsg << "[aha_laundry] Settled laundry in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()
            << " s\n";
      if (!cache.empty())
        this->SaveCache(cache, settled);
    }
    for (size_t i = 0; i < settled.size(); ++i)
    {
      this->world.SetPose(i, settled[i]);
      this->world.Garments()[i].asleep = true;
    }
    std::vector<RenderMeshSpec> specs;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
      RenderMeshSpec spec;
      spec.name = this->garments[i].id;
      spec.particle = meshes[i].renderParticle;
      for (const auto &uv : meshes[i].renderUv)
      {
        spec.uv.push_back(static_cast<float>(uv.x()));
        spec.uv.push_back(static_cast<float>(uv.y()));
      }
      for (const auto &t : meshes[i].renderTriangles)
        spec.indices.insert(spec.indices.end(), t.begin(), t.end());
      spec.albedoMap = this->garments[i].albedo;
      spec.normalMap = this->garments[i].normal;
      spec.roughness = 0.92f;
      specs.push_back(std::move(spec));
    }
    this->renderer.SetMeshes(std::move(specs));
  }

  bool LoadCache(const std::string &path, const std::vector<GarmentMesh> &meshes,
                 std::vector<std::vector<Vec3>> &out) const
  {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
      return false;
    out.clear();
    for (const auto &mesh : meshes)
    {
      uint64_t count = 0;
      stream.read(reinterpret_cast<char *>(&count), sizeof(count));
      if (!stream || count != mesh.rest2d.size())
        return false;
      std::vector<Vec3> x(count);
      stream.read(reinterpret_cast<char *>(x.data()), sizeof(Vec3) * count);
      if (!stream)
        return false;
      out.push_back(std::move(x));
    }
    return true;
  }

  void SaveCache(const std::string &path, const std::vector<std::vector<Vec3>> &x) const
  {
    std::error_code error;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path(), error);
    const std::string temporary = path + ".tmp" + std::to_string(::getpid());
    {
      std::ofstream stream(temporary, std::ios::binary);
      for (const auto &positions : x)
      {
        const uint64_t count = positions.size();
        stream.write(reinterpret_cast<const char *>(&count), sizeof(count));
        stream.write(reinterpret_cast<const char *>(positions.data()), sizeof(Vec3) * count);
      }
    }
    std::filesystem::rename(temporary, path, error);
  }

  // Crumpled piles are dropped into their containers before the episode,
  // like the referee loading the basket, the drum and the table.
  std::vector<std::vector<Vec3>> Settle(const std::vector<GarmentMesh> &meshes)
  {
    const auto &placement = this->config.value("placement", json::object());
    std::vector<std::future<std::vector<Vec3>>> piles;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
      const auto &origin = this->garments[i].origin;
      const double radius = origin == "washer" ? 0.12 : origin == "basket" ? 0.17 : 0.24;
      const uint32_t garmentSeed = static_cast<uint32_t>(this->seed * 1009 + i * 7919 + 17);
      piles.push_back(std::async(std::launch::async, [this, &meshes, i, garmentSeed, radius]()
                                 { return this->CrumplePile(meshes[i], garmentSeed, radius); }));
    }
    std::vector<std::vector<Vec3>> shapes;
    for (auto &pile : piles)
      shapes.push_back(pile.get());
    std::mt19937 rng(static_cast<uint32_t>(this->seed * 31 + 5));
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    auto place = [&](const std::vector<Vec3> &shape, const Pose &pose)
    {
      std::vector<Vec3> x;
      x.reserve(shape.size());
      for (const auto &q : shape)
        x.push_back(pose * q);
      return x;
    };
    std::vector<std::vector<Vec3>> result(meshes.size());
    std::vector<Collider> statics = this->world.Colliders();
    for (auto &c : statics)
    {
      c.movable = false;
      c.body = -1;
    }
    // Basket and table share one settling world.
    ClothWorld dry(this->parameters);
    dry.Colliders() = statics;
    std::vector<int> dryIndex(meshes.size(), -1);
    const Vec3 basketFloor = this->basketPose *
        (this->basketInteriorCenter - Vec3(0, 0, this->basketInteriorHalf.z()));
    double stack = 0.02;
    for (size_t i = 0; i < meshes.size(); ++i)
    {
      const auto &origin = this->garments[i].origin;
      if (origin == "washer")
        continue;
      Pose pose = Pose::Identity();
      pose.linear() = Eigen::AngleAxisd(2 * M_PI * unit(rng), Vec3::UnitZ()).toRotationMatrix();
      if (origin == "basket")
      {
        const Vec3 jitter(0.05 * (unit(rng) - 0.5), 0.04 * (unit(rng) - 0.5), 0);
        pose.translation() = basketFloor + this->basketPose.linear() * jitter + Vec3(0, 0, stack);
        stack += PileHeight(shapes[i]) * 0.8 + 0.03;
      }
      else
      {
        const double c = std::cos(this->tableYaw), s = std::sin(this->tableYaw);
        const json offset = placement.value("table_offset", json::array({0.0, 0.0}));
        const double u = offset.at(0).get<double>() + 0.1 * (unit(rng) - 0.5);
        const double v = offset.at(1).get<double>() + 0.06 * (unit(rng) - 0.5);
        pose.translation() = this->tableTop + Vec3(c * u - s * v, s * u + c * v, 0.01);
      }
      dryIndex[i] = dry.AddGarment(this->garments[i].id, meshes[i], Pose::Identity());
      dry.SetPose(dryIndex[i], place(shapes[i], pose));
    }
    // The drum is loaded through the door: gravity temporarily points into
    // the drum, then turns back down so that the shirts slump to the bottom.
    ClothWorld wet(this->parameters);
    std::vector<int> wetIndex(meshes.size(), -1);
    std::vector<Collider> door;
    {
      std::vector<Collider> drumStatics;
      const auto &washer = this->config.at("regions").at("washer");
      const std::string doorPrefix = washer.at("door_model").get<std::string>() + ":" +
                                     washer.value("door_link", "door");
      for (const auto &c : statics)
        (c.label.rfind(doorPrefix, 0) == 0 ? door : drumStatics).push_back(c);
      wet.Colliders() = drumStatics;
      Eigen::Matrix3d frame;
      const Vec3 up = this->drumAxis;  // out of the opening
      const Vec3 side = up.cross(Vec3::UnitZ()).normalized();
      frame.col(0) = side;
      frame.col(1) = up.cross(side);
      frame.col(2) = up;
      double depth = 0.02;
      const Vec3 back = this->drumCenter - this->drumAxis * (this->drumDepth / 2);
      for (size_t i = 0; i < meshes.size(); ++i)
      {
        if (this->garments[i].origin != "washer")
          continue;
        Pose pose = Pose::Identity();
        pose.linear() = frame * Eigen::AngleAxisd(2 * M_PI * unit(rng), Vec3::UnitZ()).toRotationMatrix();
        pose.translation() = back + up * depth;
        depth += PileHeight(shapes[i]) * 0.8 + 0.03;
        wetIndex[i] = wet.AddGarment(this->garments[i].id, meshes[i], Pose::Identity());
        wet.SetPose(wetIndex[i], place(shapes[i], pose));
      }
      wet.MutableParameters().gravity = -9.81 * up;
    }
    // The door swings shut about its hinge, sweeping protruding cloth inside.
    const Pose hinge = this->doorHinge;
    auto settleWet = std::async(std::launch::async, [this, &wet, door, hinge]()
    {
      if (wet.Garments().empty())
        return;
      // Laundry first rests at the back of the drum, then the door is shut
      // and gravity turns back down: nothing is pinched in the door.
      this->SettleGroup(wet, 3.0, 1.5);
      const size_t first = wet.Colliders().size();
      for (auto c : door)
      {
        c.movable = true;
        wet.Colliders().push_back(c);
      }
      const double open = 1.6, dt = 0.005;
      const int steps = 120;
      for (int i = 0; i <= steps; ++i)
      {
        const double angle = open * (1.0 - static_cast<double>(i) / steps);
        const Pose swing = hinge * Pose(Eigen::AngleAxisd(-angle, Vec3::UnitZ())) * hinge.inverse();
        for (size_t k = 0; k < door.size(); ++k)
        {
          wet.Colliders()[first + k].pose = swing * door[k].pose;
          if (i == 0)
            wet.Colliders()[first + k].previousPose = wet.Colliders()[first + k].pose;
        }
        for (size_t g = 0; g < wet.Garments().size(); ++g)
          wet.Wake(g);
        wet.Step(dt);
      }
      for (size_t k = 0; k < door.size(); ++k)
        wet.Colliders()[first + k].movable = false;
      wet.MutableParameters().gravity = Vec3(0, 0, -9.81);
      for (size_t g = 0; g < wet.Garments().size(); ++g)
        wet.Wake(g);
      this->SettleGroup(wet, 3.0, 0.8);
    });
    if (!dry.Garments().empty())
      this->SettleGroup(dry, 5.0, 1.2);
    settleWet.get();
    for (auto *w : {&dry, &wet})
      for (const auto &g : w->Garments())
        gzdbg << "[aha_laundry] settled " << g.id << (g.asleep ? " at rest" : " still moving")
              << ", mean speed " << g.meanSpeed << " m/s\n";
    for (size_t i = 0; i < meshes.size(); ++i)
      result[i] = dryIndex[i] >= 0 ? dry.Garments()[dryIndex[i]].x : wet.Garments()[wetIndex[i]].x;
    return result;
  }

  // ------------------------------------------------------------------ robot
  void AttachRobot(EntityComponentManager &ecm)
  {
    const auto robot = this->ModelOf(this->robotName, ecm);
    if (robot == kNullEntity)
      return;
    const auto model = gz::sim::Model(robot);
    std::vector<Collider> colliders;
    const size_t offset = this->world.Colliders().size();
    std::vector<ColliderSource> robotSources;
    std::vector<GripperState> grippers;
    std::vector<std::string> names;
    std::vector<std::pair<Entity, Entity>> fingers;
    for (const auto &entry : this->config.at("grippers"))
    {
      GripperState g;
      auto box = [](const json &b) { return Eigen::AlignedBox3d(ToVec(b.at(0)), ToVec(b.at(1))); };
      g.padA = box(entry.at("pad_a"));
      g.padB = box(entry.at("pad_b"));
      const auto a = model.LinkByName(ecm, entry.at("finger_a"));
      const auto b = model.LinkByName(ecm, entry.at("finger_b"));
      if (a == kNullEntity || b == kNullEntity)
        return;  // the robot is still being created
      fingers.push_back({a, b});
      this->padOverrides[entry.at("finger_a")] = {g.padA};
      this->padOverrides[entry.at("finger_b")] = {g.padB};
      this->fingerGripper[entry.at("finger_a")] = grippers.size();
      this->fingerGripper[entry.at("finger_b")] = grippers.size();
      g.fingerA = ToPose(gz::sim::worldPose(a, ecm));
      g.fingerB = ToPose(gz::sim::worldPose(b, ecm));
      g.previousFingerA = g.fingerA;
      grippers.push_back(g);
      names.push_back(entry.at("name"));
    }
    this->AddColliders(robot, "robot", true, ecm, colliders, robotSources);
    if (colliders.empty())
      return;
    this->world.Grippers() = grippers;
    this->gripperNames = names;
    this->fingerLinks = fingers;
    for (auto &source : robotSources)
    {
      source.index += offset;
      this->sources.push_back(source);
    }
    for (auto &c : colliders)
      this->world.Colliders().push_back(c);
    this->robotLink = robot;
    this->robotReady = true;
    gzmsg << "[aha_laundry] Robot attached: " << colliders.size() << " colliders, "
          << this->world.Grippers().size() << " grippers\n";
  }

  void UpdateColliders(EntityComponentManager &ecm)
  {
    auto &colliders = this->world.Colliders();
    std::unordered_map<Entity, Pose> poses;
    for (const auto &source : this->sources)
    {
      auto &c = colliders[source.index];
      if (!c.movable)
        continue;
      auto found = poses.find(source.link);
      if (found == poses.end())
        found = poses.emplace(source.link, ToPose(gz::sim::worldPose(source.link, ecm))).first;
      c.pose = found->second * source.local;
    }
    for (size_t g = 0; g < this->fingerLinks.size() && g < this->world.Grippers().size(); ++g)
    {
      auto &gripper = this->world.Grippers()[g];
      gripper.fingerA = ToPose(gz::sim::worldPose(this->fingerLinks[g].first, ecm));
      gripper.fingerB = ToPose(gz::sim::worldPose(this->fingerLinks[g].second, ecm));
    }
    this->basketPose = ToPose(gz::sim::worldPose(this->basketLink, ecm));
  }

  // ------------------------------------------------------------ arena door
  void OnStart(const gz::msgs::Empty &)
  {
    std::lock_guard<std::mutex> lock(this->startMutex);
    this->startRequested = true;
  }

  void ControlArenaDoor(const gz::sim::UpdateInfo &info, EntityComponentManager &ecm)
  {
    {
      std::lock_guard<std::mutex> lock(this->startMutex);
      if (this->startRequested && !this->started)
      {
        this->started = true;
        this->startTime = info.simTime;
      }
    }
    if (this->arenaDoor == kNullEntity)
      return;
    const auto &door = this->config.at("arena_door");
    const double target = door.value("open_angle", 1.5);
    const double speed = door.value("speed", 0.8);
    double angle = 0;
    if (const auto *position = ecm.Component<gz::sim::components::JointPosition>(this->arenaDoor);
        position && !position->Data().empty())
      angle = position->Data().front();
    double command = 0;
    if (this->started && std::abs(angle) < target)
      command = std::copysign(speed, target);
    ecm.SetComponentData<gz::sim::components::JointVelocityCmd>(this->arenaDoor, {command});
  }

  // ---------------------------------------------------------------- outputs
  void TrackFloorContacts()
  {
    const double limit = this->parameters.contactOffset + 0.004;
    auto &garments = this->world.Garments();
    this->floorTouched.resize(garments.size(), false);
    for (size_t i = 0; i < garments.size(); ++i)
    {
      if (garments[i].asleep || this->floorTouched[i])
        continue;
      for (const auto &p : garments[i].x)
        if (p.z() < limit && !this->InBasket(p))
        {
          this->floorTouched[i] = true;
          break;
        }
    }
  }

  void PublishRender(bool all)
  {
    auto &garments = this->world.Garments();
    if (this->renderedAsleep.size() != garments.size())
      this->renderedAsleep.assign(garments.size(), false);
    for (size_t i = 0; i < garments.size(); ++i)
    {
      const bool asleep = garments[i].asleep;
      if (!all && asleep && this->renderedAsleep[i])
        continue;
      std::vector<float> positions;
      positions.reserve(garments[i].x.size() * 3);
      for (const auto &p : garments[i].x)
      {
        positions.push_back(static_cast<float>(p.x()));
        positions.push_back(static_cast<float>(p.y()));
        positions.push_back(static_cast<float>(p.z()));
      }
      this->renderer.Publish(i, positions);
      this->renderedAsleep[i] = asleep;
    }
  }

  // The Gazebo GUI has its own scene: it shows each garment as an untextured
  // triangle marker in its main colour. GUI-only markers stay out of sensors.
  void PublishMarkers()
  {
    const auto now = std::chrono::steady_clock::now();
    if (now - this->lastGuiCheck > std::chrono::seconds(2))
    {
      std::vector<gz::transport::ServicePublisher> publishers;
      const bool listening = this->node.ServiceInfo("/marker", publishers) && !publishers.empty();
      if (listening != this->guiListening)
        gzdbg << "[aha_laundry] GUI marker service " << (listening ? "found" : "gone") << '\n';
      this->guiListening = listening;
      this->lastGuiCheck = now;
      if (!this->guiListening)
        this->markerSent.clear();
    }
    if (!this->guiListening)
      return;
    auto &garments = this->world.Garments();
    this->markerSent.resize(garments.size(), false);
    for (size_t i = 0; i < garments.size(); ++i)
    {
      if (garments[i].asleep && this->markerSent[i])
        continue;
      gz::msgs::Marker marker;
      marker.set_ns("aha_laundry");
      marker.set_id(static_cast<int>(i) + 1);
      marker.set_action(gz::msgs::Marker::ADD_MODIFY);
      marker.set_type(gz::msgs::Marker::TRIANGLE_LIST);
      marker.set_visibility(gz::msgs::Marker::GUI);
      const auto &c = this->garments[i].color;
      marker.mutable_material()->set_lighting(true);
      for (auto *color : {marker.mutable_material()->mutable_ambient(),
                          marker.mutable_material()->mutable_diffuse()})
      {
        color->set_r(c[0]);
        color->set_g(c[1]);
        color->set_b(c[2]);
        color->set_a(1.0);
      }
      const auto &x = garments[i].x;
      for (const auto &t : garments[i].mesh.triangles)
        for (const auto index : {t[0], t[1], t[2], t[0], t[2], t[1]})  // both faces
        {
          auto *point = marker.add_point();
          point->set_x(x[index].x());
          point->set_y(x[index].y());
          point->set_z(x[index].z());
        }
      // The GUI marker service answers with an empty message.
      std::function<void(const gz::msgs::Empty &, const bool)> ignore =
          [](const gz::msgs::Empty &, const bool) {};
      this->node.Request("/marker", marker, ignore);
      this->markerSent[i] = true;
    }
  }

  void PublishState(const gz::sim::UpdateInfo &info, const EntityComponentManager &ecm)
  {
    auto &garments = this->world.Garments();
    json state = {{"schema_version", 3}, {"task", "laundry"}, {"frame_id", "world"},
                  {"stamp", Stamp(info.simTime)}, {"started", this->started}};
    state["start_time"] = this->started ? json(Seconds(this->startTime)) : json(nullptr);
    if (this->robotLink != kNullEntity)
    {
      const auto robot = gz::sim::worldPose(this->ModelOfRobot(ecm), ecm);
      state["robot_pose"] = {robot.Pos().X(), robot.Pos().Y(), robot.Rot().Yaw()};
    }
    else
    {
      state["robot_pose"] = nullptr;
    }
    auto jointAngle = [&](Entity joint)
    {
      if (joint == kNullEntity)
        return json(nullptr);
      const auto *position = ecm.Component<gz::sim::components::JointPosition>(joint);
      return position && !position->Data().empty() ? json(position->Data().front()) : json(nullptr);
    };
    state["washer_door_angle"] = jointAngle(this->washerDoor);
    state["arena_door_angle"] = jointAngle(this->arenaDoor);
    const Eigen::Quaterniond q(this->basketPose.linear());
    const Vec3 &b = this->basketPose.translation();
    state["basket_pose"] = {b.x(), b.y(), b.z(), q.x(), q.y(), q.z(), q.w()};
    json grippers = json::object();
    std::vector<std::map<std::string, int>> held(garments.size());
    for (const auto &a : this->world.Attachments())
      if (a.gripper < static_cast<int>(this->gripperNames.size()))
        ++held[a.garment][this->gripperNames[a.gripper]];
    for (size_t g = 0; g < this->gripperNames.size(); ++g)
    {
      int count = 0;
      for (const auto &a : this->world.Attachments())
        count += a.gripper == static_cast<int>(g);
      const auto &gripper = this->world.Grippers()[g];
      const Vec3 centre = (gripper.fingerA * gripper.padA.center() +
                           gripper.fingerB * gripper.padB.center()) / 2;
      grippers[this->gripperNames[g]] = {{"gap", gripper.gap}, {"closed", gripper.closed},
                                         {"particles", count},
                                         {"center", {centre.x(), centre.y(), centre.z()}}};
    }
    state["grippers"] = grippers;
    this->floorTouched.resize(garments.size(), false);
    std::vector<std::optional<FoldMetrics>> folds(garments.size());
    json list = json::array();
    for (size_t i = 0; i < garments.size(); ++i)
    {
      const auto &g = garments[i];
      int washer = 0, basket = 0, table = 0, floor = 0;
      Vec3 center = Vec3::Zero();
      for (const auto &p : g.x)
      {
        center += p;
        if (this->InDrum(p))
          ++washer;
        else if (this->InBasket(p))
          ++basket;
        else if (this->OnTable(p))
          ++table;
        else if (p.z() < 0.05)
          ++floor;
      }
      const double n = static_cast<double>(g.x.size());
      center /= n;
      json contacts = json::object();
      bool supported = false;
      for (const auto &[label, count] : g.contacts)
      {
        const auto name = label.substr(0, label.find(':'));
        contacts[name] = contacts.value(name, 0) + count;
        supported = supported || name != "robot";
      }
      json held_by = json::object();
      for (const auto &[name, count] : held[i])
        held_by[name] = count;
      json entry = {{"id", g.id}, {"origin", this->garments[i].origin},
                    {"fractions", {{"washer", washer / n}, {"basket", basket / n},
                                   {"table", table / n}, {"floor", floor / n}}},
                    {"touched_floor", static_cast<bool>(this->floorTouched[i])},
                    {"held_by", held_by}, {"supported", supported || g.clothContacts > 0},
                    {"contacts", contacts}, {"cloth_contacts", g.clothContacts},
                    {"speed", g.asleep ? 0.0 : g.meanSpeed}, {"asleep", g.asleep},
                    {"center", {center.x(), center.y(), center.z()}}};
      if (table / n >= 0.9 && held[i].empty())
      {
        const auto m = ComputeFoldMetrics(g, this->tableTop.z(), this->parameters.thickness);
        folds[i] = m;
        entry["fold"] = {{"area_ratio", m.areaRatio}, {"length", m.length}, {"width", m.width},
                         {"length_ratio", m.lengthRatio}, {"width_ratio", m.widthRatio},
                         {"rectangularity", m.rectangularity}, {"height_p95", m.heightP95},
                         {"height_max", m.heightMax}, {"mean_layers", m.meanLayers},
                         {"layer_cv", m.layerCv}, {"folded_fraction", m.foldedFraction},
                         {"expected_height", m.expectedHeight},
                         {"center", {m.center.x(), m.center.y()}}, {"yaw", m.yaw}};
      }
      else
      {
        entry["fold"] = nullptr;
      }
      list.push_back(entry);
      this->floorTouched[i] = false;
    }
    state["garments"] = list;
    json stacks = json::array();
    for (size_t a = 0; a < garments.size(); ++a)
      for (size_t b = 0; b < garments.size(); ++b)
        if (a != b && folds[a] && folds[b])
        {
          const double overlap = FootprintOverlap(*folds[a], *folds[b]);
          // b lies on a: overlapping footprints with b higher.
          double za = 0, zb = 0;
          for (const auto &p : garments[a].x) za += p.z();
          for (const auto &p : garments[b].x) zb += p.z();
          za /= garments[a].x.size();
          zb /= garments[b].x.size();
          if (overlap > 0.3 && zb > za)
            stacks.push_back({{"below", garments[a].id}, {"above", garments[b].id},
                              {"overlap", overlap}});
        }
    state["stacks"] = stacks;
    gz::msgs::StringMsg message;
    message.set_data(state.dump());
    this->statePublisher.Publish(message);
    gz::msgs::Clock clock;
    clock.mutable_sim()->set_sec(state["stamp"]["sec"].get<int64_t>());
    clock.mutable_sim()->set_nsec(state["stamp"]["nanosec"].get<int32_t>());
    this->clockPublisher.Publish(clock);
  }

  Entity ModelOfRobot(const EntityComponentManager &ecm) const
  {
    return ecm.EntityByComponents(gz::sim::components::Model(),
                                  gz::sim::components::Name(this->robotName));
  }

  json scene, config, friction;
  bool configured = false, initialized = false, robotReady = false;
  ClothParameters parameters;
  ClothWorld world;
  ClothRenderer renderer;
  double spacing = 0.035, clothStep = 0.005;
  double closeGap = 0.013, openGap = 0.022, pinchMargin = 0.006;
  int seed = 0;
  std::string robotName;
  std::vector<GarmentConfig> garments;
  std::vector<ColliderSource> sources;
  std::map<std::string, std::vector<Eigen::AlignedBox3d>> padOverrides;
  std::map<std::string, int> fingerGripper;
  std::vector<std::pair<Entity, Entity>> fingerLinks;
  std::vector<std::string> gripperNames;
  Entity washerBody = kNullEntity, washerDoor = kNullEntity, basketLink = kNullEntity;
  Entity tableLink = kNullEntity, arenaDoor = kNullEntity, robotLink = kNullEntity;
  Vec3 drumCenter = Vec3::Zero(), drumAxis = Vec3::UnitX();
  double drumRadius = 0.23, drumDepth = 0.4;
  Vec3 basketInteriorCenter = Vec3::Zero(), basketInteriorHalf = Vec3::Zero();
  Pose basketPose = Pose::Identity();
  Pose doorHinge = Pose::Identity();
  Vec3 tableTop = Vec3::Zero();
  double tableYaw = 0;
  Vec2 tableHalf{0.6, 0.3};
  std::pair<Vec3, Vec3> basketWrench{Vec3::Zero(), Vec3::Zero()};
  double maxBasketForce = 40.0, maxBasketTorque = 6.0;
  std::vector<bool> floorTouched, renderedAsleep;
  bool clothChanged = false;
  std::chrono::steady_clock::duration lastCloth{}, lastRender{}, lastState{}, lastReport{};
  std::chrono::steady_clock::duration lastMarkers{};
  std::chrono::steady_clock::time_point lastGuiCheck{};
  bool guiListening = false;
  std::vector<bool> markerSent;
  std::chrono::steady_clock::duration clothWall{};
  std::mutex startMutex;
  bool startRequested = false, started = false;
  std::chrono::steady_clock::duration startTime{};
  gz::common::ConnectionPtr preRender, teardown;
  gz::transport::Node node;
  gz::transport::Node::Publisher statePublisher =
      this->node.Advertise<gz::msgs::StringMsg>("/evaluation/state");
  gz::transport::Node::Publisher clockPublisher =
      this->node.Advertise<gz::msgs::Clock>("/evaluation/clock");
};
}  // namespace aha_laundry

GZ_ADD_PLUGIN(aha_laundry::LaundrySystem, gz::sim::System,
              aha_laundry::LaundrySystem::ISystemConfigure,
              aha_laundry::LaundrySystem::ISystemPreUpdate,
              aha_laundry::LaundrySystem::ISystemPostUpdate)

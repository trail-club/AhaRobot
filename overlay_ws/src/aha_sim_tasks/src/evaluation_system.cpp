// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <gz/common/Console.hh>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/Joint.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>
#include <nlohmann/json.hpp>

#include "finger_contacts.hh"

namespace aha_sim_tasks
{
// Passive scoring instrumentation: poses and the preceding contact window form
// one atomic sample. Physics alone moves the object.
class EvaluationSystem : public gz::sim::System,
                    public gz::sim::ISystemConfigure,
                    public gz::sim::ISystemPreUpdate,
                    public gz::sim::ISystemPostUpdate
{
 public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &sdf,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override
  {
    this->configured = true;
    this->truthEnabled = sdf->Get<bool>("ground_truth", false).first;
    this->destinationName = sdf->Get<std::string>("destination_model", "").first;
    for (const auto &[key, value] : {
        std::pair{"robot_model", &this->robotName},
        std::pair{"object_model", &this->objectName},
        std::pair{"object_link", &this->objectLinkName},
        std::pair{"object_collision", &this->objectCollisionName},
        std::pair{"right_finger_link", &this->rightFingerName},
        std::pair{"left_finger_link", &this->leftFingerName}})
    {
      if (sdf->HasElement(key))
        *value = sdf->Get<std::string>(key);
      if (value->empty())
      {
        gzerr << "[aha_task_evaluation] Missing plugin parameter: " << key << '\n';
        this->configured = false;
      }
    }
  }

  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override
  {
    using namespace gz::sim;
    if (info.paused || !this->configured || this->collision != kNullEntity)
      return;
    this->robot = ecm.EntityByComponents(components::Model(), components::Name(this->robotName));
    const Entity apple = ecm.EntityByComponents(components::Model(), components::Name(this->objectName));
    this->object = Model(apple).LinkByName(ecm, this->objectLinkName);
    Model robotModel(this->robot);
    this->rightFinger = robotModel.LinkByName(ecm, this->rightFingerName);
    this->leftFinger = robotModel.LinkByName(ecm, this->leftFingerName);
    std::vector<std::string> missing;
    for (const auto &[path, entity] : {
        std::pair{this->robotName, this->robot}, std::pair{this->objectName, apple},
        std::pair{this->objectName + "/" + this->objectLinkName, this->object},
        std::pair{this->robotName + "/" + this->rightFingerName, this->rightFinger},
        std::pair{this->robotName + "/" + this->leftFingerName, this->leftFinger}})
      if (entity == kNullEntity)
        missing.push_back(path);
    const auto collisions = ecm.ChildrenByComponents(this->object, components::Collision());
    if (collisions.empty())
      missing.push_back(this->objectName + "/" + this->objectLinkName + "/" + this->objectCollisionName);
    this->rightCollisions.clear();
    this->leftCollisions.clear();
    for (const auto &[link, ids, name] : {
        std::tuple{this->rightFinger, &this->rightCollisions, this->rightFingerName},
        std::tuple{this->leftFinger, &this->leftCollisions, this->leftFingerName}})
    {
      for (const auto entity : ecm.ChildrenByComponents(link, components::Collision()))
        ids->insert(entity);
      if (link != kNullEntity && ids->empty())
        missing.push_back(this->robotName + "/" + name + " (no collision entities)");
    }
    this->destinationCollisions.clear();
    if (!this->destinationName.empty())
    {
      const auto destination = ecm.EntityByComponents(
          components::Model(), components::Name(this->destinationName));
      if (destination == kNullEntity)
      {
        missing.push_back(this->destinationName);
      }
      else
      {
        const auto top = Model(destination).LinkByName(ecm, "top");
        if (top == kNullEntity)
          missing.push_back(this->destinationName + "/top");
        else
        {
          for (const auto entity : ecm.ChildrenByComponents(top, components::Collision()))
            this->destinationCollisions.insert(entity);
          if (this->destinationCollisions.empty())
            missing.push_back(this->destinationName + "/top (no collision entities)");
        }
      }
    }
    if (!missing.empty())
    {
      const auto now = std::chrono::steady_clock::now();
      if (missing != this->lastMissing || now - this->lastDiagnostic >= std::chrono::seconds(5))
      {
        std::ostringstream names;
        for (const auto &name : missing)
          names << " " << name;
        gzerr << "[aha_task_evaluation] Unresolved entities:" << names.str()
              << "; scoring samples are unavailable\n";
        this->lastMissing = missing;
        this->lastDiagnostic = now;
      }
      return;
    }
    this->collision = collisions.front();
    this->objectCollisions = collisions;
    if (this->truthEnabled)
    {
      for (const auto entity : ecm.ChildrenByComponents(this->robot, components::Joint()))
      {
        this->joints.emplace_back(ecm.Component<components::Name>(entity)->Data(), entity);
        if (!ecm.Component<components::JointPosition>(entity))
          ecm.CreateComponent(entity, components::JointPosition());
      }
    }
    // The Physics system fills and clears this component on every step.
    for (const auto entity : this->objectCollisions)
      if (!ecm.Component<components::ContactSensorData>(entity))
        ecm.CreateComponent(entity, components::ContactSensorData());
    this->lastPublish = info.simTime;
    gzmsg << "[aha_task_evaluation] Resolved object and both finger collision sets\n";
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused || this->collision == gz::sim::kNullEntity)
      return;
    if (info.simTime < this->lastPublish)
    {
      this->contactWindow.TakeCount();
      this->destinationContact = false;
      this->simultaneousGrasp = false;
      this->lastPublish = info.simTime;
    }
    unsigned int stepMask = 0;
    for (const auto entity : this->objectCollisions)
    {
      const auto *contacts = ecm.Component<gz::sim::components::ContactSensorData>(entity);
      if (!contacts)
        continue;
      this->contactWindow.Add(contacts->Data(), entity,
                             this->rightCollisions, this->leftCollisions);
      stepMask |= FingerContactMask(contacts->Data(), entity,
                                   this->rightCollisions, this->leftCollisions);
      for (const auto &contact : contacts->Data().contact())
      {
        const auto first = contact.collision1().id();
        const auto second = contact.collision2().id();
        if ((first == entity && this->destinationCollisions.count(second)) ||
            (second == entity && this->destinationCollisions.count(first)))
          this->destinationContact = true;
      }
    }
    this->simultaneousGrasp = this->simultaneousGrasp || stepMask == 3u;
    if (info.simTime - this->lastPublish < std::chrono::milliseconds(50))
      return;
    const auto stamp = [](auto time)
    {
      const auto sec = std::chrono::duration_cast<std::chrono::seconds>(time);
      const auto nsec = std::chrono::duration_cast<std::chrono::nanoseconds>(time - sec);
      return nlohmann::json{{"sec", sec.count()}, {"nanosec", nsec.count()}};
    };
    const auto robotPose = gz::sim::worldPose(this->robot, ecm);
    const auto objectPose = gz::sim::worldPose(this->object, ecm);
    const auto end = stamp(info.simTime);
    nlohmann::json sample = {
        {"schema_version", 2}, {"frame_id", "world"}, {"stamp", end},
        {"contact_window_start", stamp(this->lastPublish)},
        {"robot_pose", {robotPose.Pos().X(), robotPose.Pos().Y(), robotPose.Rot().Yaw()}},
        {"object_position", {objectPose.Pos().X(), objectPose.Pos().Y(), objectPose.Pos().Z()}},
        {"object_orientation", {objectPose.Rot().X(), objectPose.Rot().Y(),
                                objectPose.Rot().Z(), objectPose.Rot().W()}},
        {"destination_contact", this->destinationContact},
        {"simultaneous_grasp", this->simultaneousGrasp},
        {"finger_contacts", this->contactWindow.TakeCount()}};
    if (this->truthEnabled)
    {
      nlohmann::json positions = nlohmann::json::object();
      for (const auto &[name, entity] : this->joints)
      {
        const auto *position = ecm.Component<gz::sim::components::JointPosition>(entity);
        if (position && !position->Data().empty())
          positions[name] = position->Data().front();
      }
      sample["ground_truth"] = {
          {"robot_world_pose", {robotPose.Pos().X(), robotPose.Pos().Y(), robotPose.Pos().Z(),
             robotPose.Rot().X(), robotPose.Rot().Y(), robotPose.Rot().Z(), robotPose.Rot().W()}},
          {"joint_positions", positions}};
    }
    gz::msgs::StringMsg state;
    state.set_data(sample.dump());
    this->statePublisher.Publish(state);
    // /clock can arrive separately; scoring always uses the atomic state's stamp.
    gz::msgs::Clock clock;
    clock.mutable_sim()->set_sec(end.at("sec").get<int64_t>());
    clock.mutable_sim()->set_nsec(end.at("nanosec").get<int32_t>());
    this->clockPublisher.Publish(clock);
    this->lastPublish = info.simTime;
    this->destinationContact = false;
    this->simultaneousGrasp = false;
  }

 private:
  gz::transport::Node node;
  gz::transport::Node::Publisher statePublisher =
      this->node.Advertise<gz::msgs::StringMsg>("/evaluation/state");
  gz::transport::Node::Publisher clockPublisher =
      this->node.Advertise<gz::msgs::Clock>("/evaluation/clock");
  bool configured = false;
  bool truthEnabled = false;
  std::vector<std::pair<std::string, gz::sim::Entity>> joints;
  std::string robotName, objectName, objectLinkName, objectCollisionName;
  std::string rightFingerName, leftFingerName;
  std::string destinationName;
  std::vector<gz::sim::Entity> objectCollisions;
  std::unordered_set<gz::sim::Entity> destinationCollisions;
  bool destinationContact = false;
  bool simultaneousGrasp = false;
  std::vector<std::string> lastMissing;
  std::chrono::steady_clock::time_point lastDiagnostic{};
  gz::sim::Entity robot = gz::sim::kNullEntity;
  gz::sim::Entity object = gz::sim::kNullEntity;
  gz::sim::Entity collision = gz::sim::kNullEntity;
  gz::sim::Entity rightFinger = gz::sim::kNullEntity;
  gz::sim::Entity leftFinger = gz::sim::kNullEntity;
  std::unordered_set<gz::sim::Entity> rightCollisions;
  std::unordered_set<gz::sim::Entity> leftCollisions;
  FingerContactWindow contactWindow;
  std::chrono::steady_clock::duration lastPublish{};
};
}

GZ_ADD_PLUGIN(aha_sim_tasks::EvaluationSystem, gz::sim::System,
              aha_sim_tasks::EvaluationSystem::ISystemConfigure,
              aha_sim_tasks::EvaluationSystem::ISystemPreUpdate,
              aha_sim_tasks::EvaluationSystem::ISystemPostUpdate)

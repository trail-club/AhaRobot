// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <utility>
#include <gz/msgs/uint32.pb.h>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

#include "finger_contacts.hh"

namespace aha_sim_tasks
{
// Passive scoring instrumentation. Physics alone moves the apple; this system
// only requests contact measurements and publishes them with world poses.
class EvaluationSystem : public gz::sim::System,
                    public gz::sim::ISystemPreUpdate,
                    public gz::sim::ISystemPostUpdate
{
 public:
  void PreUpdate(const gz::sim::UpdateInfo &info,
                 gz::sim::EntityComponentManager &ecm) override
  {
    using namespace gz::sim;
    if (info.paused)
      return;
    if (this->collision != kNullEntity)
      return;
    this->robot = ecm.EntityByComponents(components::Model(),
                                        components::Name("aha_robot"));
    const Entity apple = ecm.EntityByComponents(components::Model(),
                                               components::Name("apple"));
    if (this->robot == kNullEntity || apple == kNullEntity)
      return;
    Model robotModel(this->robot);
    this->object = Model(apple).LinkByName(ecm, "link");
    this->rightFinger = robotModel.LinkByName(ecm, "link_r7r");
    this->leftFinger = robotModel.LinkByName(ecm, "link_r7l");
    if (this->object == kNullEntity || this->rightFinger == kNullEntity ||
        this->leftFinger == kNullEntity)
      return;
    const auto collisions = ecm.ChildrenByComponents(
        this->object, components::Collision(), components::Name("fruit_collision"));
    if (collisions.empty())
      return;
    this->collision = collisions.front();
    for (const auto link : {this->rightFinger, this->leftFinger})
    {
      auto &ids = link == this->rightFinger ? this->rightCollisions : this->leftCollisions;
      for (const auto entity : ecm.ChildrenByComponents(link, components::Collision()))
        ids.insert(entity);
    }
    // The Physics system fills and clears this component on every step.
    ecm.CreateComponent(this->collision, components::ContactSensorData());
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused || this->robot == gz::sim::kNullEntity ||
        this->collision == gz::sim::kNullEntity)
      return;
    if (info.simTime - this->lastPublish >= std::chrono::milliseconds(50))
    {
      const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(info.simTime);
      const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime - seconds);
      // Keep the evaluation clock and ground-truth sample timestamps aligned.
      // Publishing every physics iteration can overwhelm the ROS bridge on CPUs.
      gz::msgs::Clock clock;
      clock.mutable_sim()->set_sec(seconds.count());
      clock.mutable_sim()->set_nsec(nanoseconds.count());
      this->clockPublisher.Publish(clock);
      const auto *contacts = ecm.Component<gz::sim::components::ContactSensorData>(
          this->collision);
      gz::msgs::UInt32 state;
      state.set_data(contacts ? FingerContactCount(contacts->Data(), this->collision,
          this->rightCollisions, this->leftCollisions) : 0);
      this->publisher.Publish(state);
      gz::msgs::Pose_V poses;
      for (const auto &[name, entity] :
           {std::pair{"aha_robot", this->robot}, std::pair{"apple", this->object}})
      {
        auto *pose = poses.add_pose();
        gz::msgs::Set(pose, gz::sim::worldPose(entity, ecm));
        pose->set_name(name);
        auto *header = pose->mutable_header();
        header->mutable_stamp()->set_sec(seconds.count());
        header->mutable_stamp()->set_nsec(nanoseconds.count());
        auto *parent = header->add_data();
        parent->set_key("frame_id");
        parent->add_value("world");
        auto *child = header->add_data();
        child->set_key("child_frame_id");
        child->add_value(name);
      }
      this->posePublisher.Publish(poses);
      this->lastPublish = info.simTime;
    }
  }

 private:
  gz::transport::Node node;
  gz::transport::Node::Publisher publisher =
      this->node.Advertise<gz::msgs::UInt32>("/evaluation/finger_contacts");
  gz::transport::Node::Publisher posePublisher =
      this->node.Advertise<gz::msgs::Pose_V>("/evaluation/poses");
  gz::transport::Node::Publisher clockPublisher =
      this->node.Advertise<gz::msgs::Clock>("/evaluation/clock");
  gz::sim::Entity robot = gz::sim::kNullEntity;
  gz::sim::Entity object = gz::sim::kNullEntity;
  gz::sim::Entity collision = gz::sim::kNullEntity;
  gz::sim::Entity rightFinger = gz::sim::kNullEntity;
  gz::sim::Entity leftFinger = gz::sim::kNullEntity;
  std::unordered_set<gz::sim::Entity> rightCollisions;
  std::unordered_set<gz::sim::Entity> leftCollisions;
  std::chrono::steady_clock::duration lastPublish{};
};
}

GZ_ADD_PLUGIN(aha_sim_tasks::EvaluationSystem, gz::sim::System,
              aha_sim_tasks::EvaluationSystem::ISystemPreUpdate,
              aha_sim_tasks::EvaluationSystem::ISystemPostUpdate)

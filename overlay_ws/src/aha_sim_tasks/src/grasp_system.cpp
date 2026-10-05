// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <utility>
#include <gz/math/Pose3.hh>
#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/clock.pb.h>
#include <gz/msgs/pose_v.pb.h>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Joint.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/JointPosition.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/transport/Node.hh>

namespace aha_sim_tasks
{
// A proximity grasp abstraction, driven by measured finger positions. Starts
// released and never teleports the object. The policy cannot command the latch.
class GraspSystem : public gz::sim::System,
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
    if (this->robot == kNullEntity)
    {
      this->robot = ecm.EntityByComponents(components::Model(),
                                           components::Name("aha_robot"));
      if (this->robot == kNullEntity)
        return;
      Model model(this->robot);
      this->gripper = model.LinkByName(ecm, "link_r6");
      this->rightFinger = model.JointByName(ecm, "joint_r7r");
      this->leftFinger = model.JointByName(ecm, "joint_r7l");
      Joint(this->rightFinger).EnablePositionCheck(ecm);
      Joint(this->leftFinger).EnablePositionCheck(ecm);
      const Entity apple = ecm.EntityByComponents(components::Model(),
                                                 components::Name("apple"));
      this->object = Model(apple).LinkByName(ecm, "link");
    }
    if (this->gripper == kNullEntity || this->object == kNullEntity)
      return;
    const auto *right = ecm.Component<components::JointPosition>(this->rightFinger);
    const auto *left = ecm.Component<components::JointPosition>(this->leftFinger);
    if (!right || !left || right->Data().empty() || left->Data().empty())
      return;
    const bool open = right->Data()[0] >= 0.055 && left->Data()[0] <= -0.055;
    const bool closed = right->Data()[0] <= 0.047 && left->Data()[0] >= -0.047;
    if (open)
    {
      this->opened = true;
      if (this->joint != kNullEntity)
      {
        ecm.RequestRemoveEntity(this->joint);
        this->joint = kNullEntity;
      }
    }
    const auto grasp = worldPose(this->gripper, ecm) *
                       gz::math::Pose3d(0.063, 0, 0.195, 0, 0, 0);
    const double distance = grasp.Pos().Distance(worldPose(this->object, ecm).Pos());
    if (this->opened && closed && distance <= 0.055 && this->joint == kNullEntity)
    {
      this->joint = ecm.CreateEntity();
      ecm.CreateComponent(this->joint,
          components::DetachableJoint({this->gripper, this->object, "fixed"}));
      this->opened = false;
    }
  }

  void PostUpdate(const gz::sim::UpdateInfo &info,
                  const gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused || this->robot == gz::sim::kNullEntity ||
        this->object == gz::sim::kNullEntity)
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
      gz::msgs::Boolean state;
      state.set_data(this->joint != gz::sim::kNullEntity);
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
      this->node.Advertise<gz::msgs::Boolean>("/evaluation/grasped");
  gz::transport::Node::Publisher posePublisher =
      this->node.Advertise<gz::msgs::Pose_V>("/evaluation/poses");
  gz::transport::Node::Publisher clockPublisher =
      this->node.Advertise<gz::msgs::Clock>("/evaluation/clock");
  gz::sim::Entity robot = gz::sim::kNullEntity;
  gz::sim::Entity gripper = gz::sim::kNullEntity;
  gz::sim::Entity object = gz::sim::kNullEntity;
  gz::sim::Entity rightFinger = gz::sim::kNullEntity;
  gz::sim::Entity leftFinger = gz::sim::kNullEntity;
  gz::sim::Entity joint = gz::sim::kNullEntity;
  bool opened = false;
  std::chrono::steady_clock::duration lastPublish{};
};
}

GZ_ADD_PLUGIN(aha_sim_tasks::GraspSystem, gz::sim::System,
              aha_sim_tasks::GraspSystem::ISystemPreUpdate,
              aha_sim_tasks::GraspSystem::ISystemPostUpdate)

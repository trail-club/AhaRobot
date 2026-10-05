// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <gtest/gtest.h>
#include <gz/sim/Model.hh>
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/TestFixture.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Collision.hh>
#include <gz/sim/components/ContactSensorData.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>

TEST(PassiveEvaluation, NearbyUnsupportedAppleFallsUnderGravity)
{
  // Apple starts at the old proximity-latch grasp point, with neither finger
  // touching it. Loading scoring instrumentation must not support the apple.
  const std::string world = R"(
    <sdf version="1.10"><world name="free_apple">
      <physics name="1ms" type="ignored"><max_step_size>0.001</max_step_size></physics>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename=")" + std::string(EVALUATION_PLUGIN_PATH) + R"("
              name="aha_sim_tasks::EvaluationSystem"/>
      <model name="aha_robot"><static>true</static>
        <link name="link_r6"><pose>0 0 1 0 0 0</pose></link>
        <link name="link_r7r"><pose>0.063 0.08 1.195 0 0 0</pose>
          <collision name="finger"><geometry><sphere><radius>0.01</radius></sphere></geometry></collision>
        </link>
        <link name="link_r7l"><pose>0.063 -0.08 1.195 0 0 0</pose>
          <collision name="finger"><geometry><sphere><radius>0.01</radius></sphere></geometry></collision>
        </link>
        <joint name="right" type="fixed"><parent>link_r6</parent><child>link_r7r</child></joint>
        <joint name="left" type="fixed"><parent>link_r6</parent><child>link_r7l</child></joint>
      </model>
      <model name="apple"><pose>0.063 0 1.195 0 0 0</pose><link name="link">
        <inertial><mass>0.08</mass><inertia>
          <ixx>0.0000512</ixx><iyy>0.0000512</iyy><izz>0.0000512</izz>
        </inertia></inertial>
        <collision name="fruit_collision"><geometry><sphere><radius>0.04</radius></sphere></geometry></collision>
      </link></model>
      <model name="floor"><static>true</static><link name="link">
        <collision name="floor"><geometry><plane><normal>0 0 1</normal><size>5 5</size></plane></geometry></collision>
      </link></model>
    </world></sdf>)";
  gz::sim::ServerConfig config;
  ASSERT_TRUE(config.SetSdfString(world));
  gz::sim::TestFixture fixture(config);
  double finalHeight = 1.195;
  bool contactMeasurementsEnabled = false;
  unsigned int attachedJoints = 0;
  fixture.OnPostUpdate([&](const gz::sim::UpdateInfo &,
                           const gz::sim::EntityComponentManager &ecm)
  {
    using namespace gz::sim;
    const auto apple = ecm.EntityByComponents(components::Model(), components::Name("apple"));
    ASSERT_NE(kNullEntity, apple);
    const auto link = Model(apple).LinkByName(ecm, "link");
    finalHeight = worldPose(link, ecm).Pos().Z();
    const auto collisions = ecm.ChildrenByComponents(link,
        components::Collision(), components::Name("fruit_collision"));
    ASSERT_EQ(1u, collisions.size());
    contactMeasurementsEnabled = ecm.Component<components::ContactSensorData>(
        collisions.front()) != nullptr;
    ecm.Each<components::DetachableJoint>([&](const Entity &,
        const components::DetachableJoint *)
    {
      ++attachedJoints;
      return true;
    });
  }).Finalize();
  ASSERT_TRUE(fixture.Server()->Run(true, 1000, false));
  EXPECT_TRUE(contactMeasurementsEnabled);
  EXPECT_EQ(0u, attachedJoints);
  EXPECT_NEAR(0.04, finalHeight, 0.005);
}

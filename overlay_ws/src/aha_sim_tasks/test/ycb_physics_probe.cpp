// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <gz/sim/Link.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Server.hh>
#include <gz/sim/ServerConfig.hh>
#include <gz/sim/TestFixture.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <nlohmann/json.hpp>

int main(int argc, char **argv)
{
  if (argc != 2)
    return 2;
  gz::sim::ServerConfig config;
  config.SetSdfFile(argv[1]);
  gz::sim::TestFixture fixture(config);
  nlohmann::json result;
  fixture.OnPreUpdate([&](const gz::sim::UpdateInfo &info,
                         gz::sim::EntityComponentManager &ecm)
  {
    using namespace gz::sim;
    const auto entity = ecm.EntityByComponents(components::Model(), components::Name("apple"));
    if (entity == kNullEntity)
      return;
    Model model(entity);
    Link link(model.LinkByName(ecm, "link"));
    if (info.iterations >= 1500 && info.iterations < 1510)
    {
      const auto mass = link.WorldInertial(ecm)->MassMatrix().Mass();
      link.AddWorldWrench(ecm, {mass * 12, 0, 0}, {0, 0, 0});
    }
    if (info.iterations == 2500)
    {
      model.SetWorldPoseCmd(ecm, gz::math::Pose3d(0, 0, 1.2, 0, 0, 0));
    }
  }).OnPostUpdate([&](const gz::sim::UpdateInfo &info,
                     const gz::sim::EntityComponentManager &ecm)
  {
    using namespace gz::sim;
    const auto entity = ecm.EntityByComponents(components::Model(), components::Name("apple"));
    if (entity == kNullEntity)
      return;
    const auto pose = worldPose(Model(entity).LinkByName(ecm, "link"), ecm);
    for (const auto &[iteration, name] : {std::pair{1499u, "rest"}, std::pair{1510u, "impulse"},
         std::pair{2499u, "pushed"}, std::pair{3999u, "dropped"}})
      if (info.iterations == iteration)
        result[name] = {pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z()};
  }).Finalize();
  if (!fixture.Server()->Run(true, 4000, false))
    return 1;
  std::cout << "YCB_PHYSICS=" << result.dump() << '\n';
  return result.size() == 4 ? 0 : 1;
}

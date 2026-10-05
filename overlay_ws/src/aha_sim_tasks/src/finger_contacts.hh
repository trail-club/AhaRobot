// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#ifndef AHA_SIM_TASKS_FINGER_CONTACTS_HH_
#define AHA_SIM_TASKS_FINGER_CONTACTS_HH_

#include <unordered_set>
#include <gz/msgs/contacts.pb.h>
#include <gz/sim/Entity.hh>

namespace aha_sim_tasks
{
// Count distinct fingers touching this object in the current physics step.
// No history, proximity threshold, or commanded finger position is involved.
inline unsigned int FingerContactCount(const gz::msgs::Contacts &contacts,
    gz::sim::Entity objectCollision,
    const std::unordered_set<gz::sim::Entity> &rightCollisions,
    const std::unordered_set<gz::sim::Entity> &leftCollisions)
{
  bool right = false;
  bool left = false;
  for (const auto &contact : contacts.contact())
  {
    const auto first = contact.collision1().id();
    const auto second = contact.collision2().id();
    if (first != objectCollision && second != objectCollision)
      continue;
    const auto other = first == objectCollision ? second : first;
    right = right || rightCollisions.count(other) != 0;
    left = left || leftCollisions.count(other) != 0;
  }
  return static_cast<unsigned int>(right) + static_cast<unsigned int>(left);
}
}

#endif

// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#ifndef AHA_SIM_TASKS_FINGER_CONTACTS_HH_
#define AHA_SIM_TASKS_FINGER_CONTACTS_HH_

#include <unordered_set>
#include <gz/msgs/contacts.pb.h>
#include <gz/sim/Entity.hh>

namespace aha_sim_tasks
{
// Bits identify the fingers touching in this physics step (right=1, left=2).
inline unsigned int FingerContactMask(const gz::msgs::Contacts &contacts,
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
  return static_cast<unsigned int>(right) | (static_cast<unsigned int>(left) << 1);
}

inline unsigned int FingerContactCount(const gz::msgs::Contacts &contacts,
    gz::sim::Entity objectCollision,
    const std::unordered_set<gz::sim::Entity> &rightCollisions,
    const std::unordered_set<gz::sim::Entity> &leftCollisions)
{
  const auto mask = FingerContactMask(contacts, objectCollision, rightCollisions, leftCollisions);
  return (mask & 1u) + ((mask >> 1) & 1u);
}

// Collect distinct fingers across a publication window, not just its final step.
// TakeCount clears the window: there is no persistence into the next sample.
class FingerContactWindow
{
 public:
  void Add(const gz::msgs::Contacts &contacts, gz::sim::Entity objectCollision,
      const std::unordered_set<gz::sim::Entity> &rightCollisions,
      const std::unordered_set<gz::sim::Entity> &leftCollisions)
  {
    this->mask |= FingerContactMask(contacts, objectCollision, rightCollisions, leftCollisions);
  }

  unsigned int TakeCount()
  {
    const auto count = (this->mask & 1u) + ((this->mask >> 1) & 1u);
    this->mask = 0;
    return count;
  }

 private:
  unsigned int mask = 0;
};
}

#endif

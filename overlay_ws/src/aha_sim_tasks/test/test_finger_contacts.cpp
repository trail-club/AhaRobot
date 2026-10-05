// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "finger_contacts.hh"

namespace
{
void Contact(gz::msgs::Contacts &contacts, unsigned int first, unsigned int second)
{
  auto *contact = contacts.add_contact();
  contact->mutable_collision1()->set_id(first);
  contact->mutable_collision2()->set_id(second);
}

unsigned int Count(const gz::msgs::Contacts &contacts)
{
  return aha_sim_tasks::FingerContactCount(contacts, 1, {2, 3}, {4});
}

TEST(FingerContacts, TableAndOtherObjectContactsDoNotCount)
{
  gz::msgs::Contacts contacts;
  Contact(contacts, 1, 5);  // Apple touching table.
  Contact(contacts, 2, 5);  // Finger touching something other than apple.
  Contact(contacts, 4, 2);  // Fingers touching each other.
  EXPECT_EQ(0u, Count(contacts));
}

TEST(FingerContacts, MultipleContactPointsOnOneFingerAreOneContact)
{
  gz::msgs::Contacts contacts;
  Contact(contacts, 1, 2);
  Contact(contacts, 3, 1);  // Another collision on the same finger, reversed order.
  EXPECT_EQ(1u, Count(contacts));
  Contact(contacts, 4, 1);
  EXPECT_EQ(2u, Count(contacts));
}

TEST(FingerContacts, ContactLossImmediatelyClearsEvidence)
{
  gz::msgs::Contacts contacts;
  Contact(contacts, 1, 2);
  Contact(contacts, 1, 4);
  EXPECT_EQ(2u, Count(contacts));
  contacts.Clear();
  EXPECT_EQ(0u, Count(contacts));
}

TEST(FingerContacts, WindowCollectsDistinctFingersAcrossStepsAndJitter)
{
  aha_sim_tasks::FingerContactWindow window;
  gz::msgs::Contacts contacts;
  Contact(contacts, 1, 2);
  window.Add(contacts, 1, {2, 3}, {4});
  contacts.Clear();  // A solver step with no contact must not erase the window.
  window.Add(contacts, 1, {2, 3}, {4});
  Contact(contacts, 4, 1);  // The other finger, not a second point on the first.
  window.Add(contacts, 1, {2, 3}, {4});
  contacts.Clear();
  window.Add(contacts, 1, {2, 3}, {4});
  EXPECT_EQ(2u, window.TakeCount());
  EXPECT_EQ(0u, window.TakeCount());  // Release is visible in the next window.
}

TEST(FingerContacts, WindowDoesNotDoubleCountOneFingerOrIncludeTable)
{
  aha_sim_tasks::FingerContactWindow window;
  gz::msgs::Contacts contacts;
  Contact(contacts, 1, 2);
  Contact(contacts, 3, 1);
  Contact(contacts, 1, 5);
  for (int step = 0; step < 50; ++step)
    window.Add(contacts, 1, {2, 3}, {4});
  EXPECT_EQ(1u, window.TakeCount());
  contacts.Clear();
  for (int step = 0; step < 50; ++step)
    window.Add(contacts, 1, {2, 3}, {4});
  EXPECT_EQ(0u, window.TakeCount());
}
}

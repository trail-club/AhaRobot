// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#ifndef AHA_SIM_TASKS_LAUNDRY_PATTERN_IO_HH_
#define AHA_SIM_TASKS_LAUNDRY_PATTERN_IO_HH_

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "cloth.hh"

namespace aha_laundry
{
// Reads a pattern written by aha_sim_tasks/laundry_pattern.py.
inline GarmentPattern PatternFromJson(const nlohmann::json &json)
{
  GarmentPattern pattern;
  pattern.name = json.value("name", "garment");
  pattern.mass = json.at("mass").get<double>();
  for (auto [key, panel] : {std::pair{"front", &pattern.front}, std::pair{"back", &pattern.back}})
  {
    const auto &entry = json.at(key);
    for (const auto &point : entry.at("outline"))
      panel->outline.emplace_back(point.at(0).get<double>(), point.at(1).get<double>());
    for (const auto &seam : entry.at("seam"))
      panel->seam.push_back(seam.get<bool>());
    const auto &uv = entry.at("uv");
    panel->uvOffset = Vec2(uv.at("offset").at(0), uv.at("offset").at(1));
    panel->uvScale = Vec2(uv.at("scale").at(0), uv.at("scale").at(1));
  }
  return pattern;
}

inline GarmentPattern LoadPattern(const std::string &path)
{
  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Cannot read garment pattern " + path);
  return PatternFromJson(nlohmann::json::parse(stream));
}
}  // namespace aha_laundry

#endif

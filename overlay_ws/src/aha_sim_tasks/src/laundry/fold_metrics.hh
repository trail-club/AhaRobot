// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#ifndef AHA_SIM_TASKS_LAUNDRY_FOLD_METRICS_HH_
#define AHA_SIM_TASKS_LAUNDRY_FOLD_METRICS_HH_

#include <cstdint>
#include <vector>

#include "cloth.hh"

namespace aha_laundry
{
// Geometry of a garment lying on a horizontal surface, the basis of the
// folding score. Layers count the cloth sheets above each footprint cell: a
// spread T-shirt has two everywhere (front and back panels).
struct FoldMetrics
{
  double footprintArea = 0;
  double areaRatio = 0;  // footprint / flat panel area
  double length = 0;  // minimum-area rectangle around the footprint
  double width = 0;
  double rectangularity = 0;  // footprint / rectangle area
  double lengthRatio = 0;  // rectangle sides / flat pattern sides
  double widthRatio = 0;
  double heightMax = 0;  // above the surface
  double heightP95 = 0;
  double meanLayers = 0;
  double layerCv = 0;  // coefficient of variation of layers over the footprint
  double foldedFraction = 0;  // footprint cells with more than two layers
  double expectedHeight = 0;  // stack height for the mean layer count
  Vec2 center{0, 0};
  double yaw = 0;  // rectangle orientation
  std::vector<int64_t> cells;  // sorted footprint cell keys
};

FoldMetrics ComputeFoldMetrics(const Garment &garment, double surfaceZ, double thickness,
                               double cell = 0.01);

// Fraction of the smaller footprint covered by the other.
double FootprintOverlap(const FoldMetrics &a, const FoldMetrics &b);
}  // namespace aha_laundry

#endif

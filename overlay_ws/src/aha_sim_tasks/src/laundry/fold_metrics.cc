// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include "fold_metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace aha_laundry
{
namespace
{
int64_t CellKey(int64_t x, int64_t y)
{
  return (x << 32) ^ (y & 0xffffffff);
}

double Cross(const Vec2 &o, const Vec2 &a, const Vec2 &b)
{
  return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
}

std::vector<Vec2> ConvexHull(std::vector<Vec2> points)
{
  std::sort(points.begin(), points.end(), [](const Vec2 &a, const Vec2 &b)
            { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });
  if (points.size() < 3)
    return points;
  std::vector<Vec2> hull(2 * points.size());
  size_t k = 0;
  for (size_t i = 0; i < points.size(); ++i)
  {
    while (k >= 2 && Cross(hull[k - 2], hull[k - 1], points[i]) <= 0)
      --k;
    hull[k++] = points[i];
  }
  for (size_t i = points.size() - 1, t = k + 1; i > 0; --i)
  {
    while (k >= t && Cross(hull[k - 2], hull[k - 1], points[i - 1]) <= 0)
      --k;
    hull[k++] = points[i - 1];
  }
  hull.resize(k - 1);
  return hull;
}
}  // namespace

FoldMetrics ComputeFoldMetrics(const Garment &garment, double surfaceZ, double thickness,
                               double cell)
{
  FoldMetrics m;
  const auto &x = garment.x;
  std::unordered_map<int64_t, int> layers;
  for (const auto &t : garment.mesh.triangles)
  {
    const Vec2 a = x[t[0]].head<2>(), b = x[t[1]].head<2>(), c = x[t[2]].head<2>();
    const double area = Cross(a, b, c);
    if (std::abs(area) < 1e-12)
      continue;
    const Vec2 lower = a.cwiseMin(b).cwiseMin(c), upper = a.cwiseMax(b).cwiseMax(c);
    for (int64_t i = std::floor(lower.x() / cell); i <= std::floor(upper.x() / cell); ++i)
      for (int64_t j = std::floor(lower.y() / cell); j <= std::floor(upper.y() / cell); ++j)
      {
        const Vec2 p((i + 0.5) * cell, (j + 0.5) * cell);
        const double w0 = Cross(b, c, p), w1 = Cross(c, a, p), w2 = Cross(a, b, p);
        if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0))
          ++layers[CellKey(i, j)];
      }
  }
  if (layers.empty())
    return m;
  double sum = 0, sumSquares = 0;
  int folded = 0;
  m.cells.reserve(layers.size());
  for (const auto &[key, count] : layers)
  {
    m.cells.push_back(key);
    sum += count;
    sumSquares += static_cast<double>(count) * count;
    folded += count > 2;
  }
  std::sort(m.cells.begin(), m.cells.end());
  const double n = layers.size();
  m.footprintArea = n * cell * cell;
  m.areaRatio = m.footprintArea / std::max(garment.mesh.frontArea, 1e-9);
  m.meanLayers = sum / n;
  m.layerCv = std::sqrt(std::max(0.0, sumSquares / n - m.meanLayers * m.meanLayers)) /
              m.meanLayers;
  m.foldedFraction = folded / n;
  m.expectedHeight = thickness * std::max(0.0, m.meanLayers - 0.5);

  std::vector<double> heights;
  std::vector<Vec2> planar;
  heights.reserve(x.size());
  planar.reserve(x.size());
  for (const auto &p : x)
  {
    heights.push_back(p.z() - surfaceZ);
    planar.push_back(p.head<2>());
  }
  std::sort(heights.begin(), heights.end());
  m.heightMax = heights.back();
  m.heightP95 = heights[static_cast<size_t>(0.95 * (heights.size() - 1))];

  const auto hull = ConvexHull(planar);
  double bestArea = std::numeric_limits<double>::max();
  for (size_t i = 0; i < hull.size(); ++i)
  {
    Vec2 edge = hull[(i + 1) % hull.size()] - hull[i];
    if (edge.norm() < 1e-9)
      continue;
    edge.normalize();
    const Vec2 normal(-edge.y(), edge.x());
    double minU = 1e9, maxU = -1e9, minV = 1e9, maxV = -1e9;
    for (const auto &p : hull)
    {
      minU = std::min(minU, p.dot(edge));
      maxU = std::max(maxU, p.dot(edge));
      minV = std::min(minV, p.dot(normal));
      maxV = std::max(maxV, p.dot(normal));
    }
    const double area = (maxU - minU) * (maxV - minV);
    if (area < bestArea)
    {
      bestArea = area;
      const double u = maxU - minU, v = maxV - minV;
      m.length = std::max(u, v);
      m.width = std::min(u, v);
      m.center = edge * (minU + maxU) / 2 + normal * (minV + maxV) / 2;
      const Vec2 along = u >= v ? edge : normal;
      m.yaw = std::atan2(along.y(), along.x());
    }
  }
  if (bestArea < std::numeric_limits<double>::max() && bestArea > 0)
    m.rectangularity = std::min(1.0, m.footprintArea / bestArea);
  const double flatLength = std::max(garment.mesh.extent.x(), garment.mesh.extent.y());
  const double flatWidth = std::min(garment.mesh.extent.x(), garment.mesh.extent.y());
  m.lengthRatio = m.length / std::max(flatLength, 1e-9);
  m.widthRatio = m.width / std::max(flatWidth, 1e-9);
  return m;
}

double FootprintOverlap(const FoldMetrics &a, const FoldMetrics &b)
{
  if (a.cells.empty() || b.cells.empty())
    return 0;
  size_t common = 0;
  auto i = a.cells.begin(), j = b.cells.begin();
  while (i != a.cells.end() && j != b.cells.end())
  {
    if (*i < *j)
      ++i;
    else if (*j < *i)
      ++j;
    else
    {
      ++common;
      ++i;
      ++j;
    }
  }
  return static_cast<double>(common) / std::min(a.cells.size(), b.cells.size());
}
}  // namespace aha_laundry

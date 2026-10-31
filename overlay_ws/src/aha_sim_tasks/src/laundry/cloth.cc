// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include "cloth.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace aha_laundry
{
namespace
{
double Cross2(const Vec2 &a, const Vec2 &b)
{
  return a.x() * b.y() - a.y() * b.x();
}

bool InsidePolygon(const std::vector<Vec2> &polygon, const Vec2 &p)
{
  bool inside = false;
  for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
  {
    const Vec2 &a = polygon[i], &b = polygon[j];
    if ((a.y() > p.y()) != (b.y() > p.y()) &&
        p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y()) + a.x())
      inside = !inside;
  }
  return inside;
}

double SegmentDistance(const Vec2 &p, const Vec2 &a, const Vec2 &b)
{
  const Vec2 ab = b - a;
  const double t = std::clamp((p - a).dot(ab) / std::max(ab.squaredNorm(), 1e-18), 0.0, 1.0);
  return (a + t * ab - p).norm();
}

double BoundaryDistance(const std::vector<Vec2> &polygon, const Vec2 &p)
{
  double best = std::numeric_limits<double>::max();
  for (size_t i = 0; i < polygon.size(); ++i)
    best = std::min(best, SegmentDistance(p, polygon[i], polygon[(i + 1) % polygon.size()]));
  return best;
}

// Bowyer-Watson triangulation; adequate for the few hundred points per panel.
std::vector<std::array<uint32_t, 3>> Delaunay(const std::vector<Vec2> &points)
{
  Vec2 lower = points.front(), upper = points.front();
  for (const auto &p : points)
  {
    lower = lower.cwiseMin(p);
    upper = upper.cwiseMax(p);
  }
  const double span = std::max(upper.x() - lower.x(), upper.y() - lower.y()) * 20.0;
  const Vec2 center = (lower + upper) / 2;
  std::vector<Vec2> all = points;
  const uint32_t s0 = all.size();
  all.push_back(center + Vec2(-span, -span));
  all.push_back(center + Vec2(span, -span));
  all.push_back(center + Vec2(0, span));
  struct Tri
  {
    std::array<uint32_t, 3> v;
    Vec2 c;
    double r2;
  };
  auto make = [&](uint32_t a, uint32_t b, uint32_t c)
  {
    if (Cross2(all[b] - all[a], all[c] - all[a]) < 0)
      std::swap(b, c);
    const Vec2 &A = all[a], &B = all[b], &C = all[c];
    const double d = 2 * (A.x() * (B.y() - C.y()) + B.x() * (C.y() - A.y()) +
                          C.x() * (A.y() - B.y()));
    const double a2 = A.squaredNorm(), b2 = B.squaredNorm(), c2 = C.squaredNorm();
    const Vec2 o((a2 * (B.y() - C.y()) + b2 * (C.y() - A.y()) + c2 * (A.y() - B.y())) / d,
                 (a2 * (C.x() - B.x()) + b2 * (A.x() - C.x()) + c2 * (B.x() - A.x())) / d);
    return Tri{{a, b, c}, o, (A - o).squaredNorm()};
  };
  std::vector<Tri> triangles{make(s0, s0 + 1, s0 + 2)};
  for (uint32_t i = 0; i < points.size(); ++i)
  {
    const Vec2 &p = all[i];
    std::map<std::pair<uint32_t, uint32_t>, int> boundary;
    std::vector<Tri> kept;
    kept.reserve(triangles.size() + 2);
    for (const auto &t : triangles)
    {
      if ((p - t.c).squaredNorm() <= t.r2 * (1 + 1e-12))
      {
        for (int k = 0; k < 3; ++k)
        {
          uint32_t a = t.v[k], b = t.v[(k + 1) % 3];
          ++boundary[{std::min(a, b), std::max(a, b)}];
        }
      }
      else
      {
        kept.push_back(t);
      }
    }
    for (const auto &[edge, count] : boundary)
      if (count == 1)
        kept.push_back(make(edge.first, edge.second, i));
    triangles.swap(kept);
  }
  std::vector<std::array<uint32_t, 3>> result;
  for (const auto &t : triangles)
    if (t.v[0] < s0 && t.v[1] < s0 && t.v[2] < s0)
      result.push_back(t.v);
  return result;
}

struct PanelMesh
{
  std::vector<Vec2> points;
  std::vector<bool> onSeam;
  std::vector<std::array<uint32_t, 3>> triangles;
};

PanelMesh MeshPanel(const PanelPattern &panel, double spacing)
{
  const auto &outline = panel.outline;
  if (outline.size() < 3 || panel.seam.size() != outline.size())
    throw std::invalid_argument("Panel outline needs >= 3 vertices and one seam flag per edge");
  PanelMesh mesh;
  const size_t n = outline.size();
  for (size_t i = 0; i < n; ++i)
  {
    const Vec2 &a = outline[i], &b = outline[(i + 1) % n];
    const int samples = std::max(1, static_cast<int>(std::lround((b - a).norm() / spacing)));
    const bool previousSeam = panel.seam[(i + n - 1) % n];
    for (int k = 0; k < samples; ++k)
    {
      mesh.points.push_back(a + (b - a) * (static_cast<double>(k) / samples));
      // A corner belongs to the seam if either adjacent edge is sewn.
      mesh.onSeam.push_back(panel.seam[i] || (k == 0 && previousSeam));
    }
  }
  const size_t boundaryCount = mesh.points.size();
  Vec2 lower = outline.front(), upper = outline.front();
  for (const auto &p : outline)
  {
    lower = lower.cwiseMin(p);
    upper = upper.cwiseMax(p);
  }
  const double rowStep = spacing * std::sqrt(3.0) / 2;
  int row = 0;
  for (double y = lower.y() + rowStep / 2; y < upper.y(); y += rowStep, ++row)
  {
    for (double x = lower.x() + (row % 2 ? spacing / 2 : 0); x < upper.x(); x += spacing)
    {
      const Vec2 p(x, y);
      if (InsidePolygon(outline, p) && BoundaryDistance(outline, p) > 0.55 * spacing)
      {
        mesh.points.push_back(p);
        mesh.onSeam.push_back(false);
      }
    }
  }
  (void)boundaryCount;
  for (const auto &t : Delaunay(mesh.points))
  {
    const Vec2 &a = mesh.points[t[0]], &b = mesh.points[t[1]], &c = mesh.points[t[2]];
    if (std::abs(Cross2(b - a, c - a)) < 1e-4 * spacing * spacing)
      continue;
    const Vec2 centroid = (a + b + c) / 3;
    bool keep = InsidePolygon(outline, centroid);
    for (int k = 0; keep && k < 3; ++k)
    {
      const Vec2 mid = (mesh.points[t[k]] + mesh.points[t[(k + 1) % 3]]) / 2;
      keep = InsidePolygon(outline, mid) || BoundaryDistance(outline, mid) < 1e-6;
    }
    if (keep)
      mesh.triangles.push_back(t);
  }
  return mesh;
}

Pose Interpolate(const Pose &a, const Pose &b, double t)
{
  Pose result = Pose::Identity();
  result.translation() = a.translation() + t * (b.translation() - a.translation());
  result.linear() = Eigen::Quaterniond(a.linear())
                        .slerp(t, Eigen::Quaterniond(b.linear()))
                        .toRotationMatrix();
  return result;
}

Eigen::AlignedBox3d ColliderBounds(const Collider &c)
{
  Vec3 half;
  switch (c.type)
  {
    case ShapeType::Box: half = c.halfExtents; break;
    case ShapeType::Sphere: half = Vec3::Constant(c.radius); break;
    case ShapeType::Capsule: half = Vec3(c.radius, c.radius, c.halfLength + c.radius); break;
    case ShapeType::Cylinder: half = Vec3(c.radius, c.radius, c.halfLength); break;
    case ShapeType::Plane:
      return Eigen::AlignedBox3d(Vec3::Constant(-1e6), Vec3::Constant(1e6));
  }
  const Eigen::Matrix3d r = c.pose.linear().cwiseAbs();
  const Vec3 extent = r * half;
  Eigen::AlignedBox3d box(c.pose.translation() - extent, c.pose.translation() + extent);
  const Eigen::Matrix3d rp = c.previousPose.linear().cwiseAbs();
  const Vec3 previousExtent = rp * half;
  box.extend(Eigen::AlignedBox3d(c.previousPose.translation() - previousExtent,
                                 c.previousPose.translation() + previousExtent));
  return box;
}

double LocalDistance(const Collider &c, const Vec3 &p, Vec3 *n)
{
  switch (c.type)
  {
    case ShapeType::Box:
    {
      const Vec3 q = p.cwiseAbs() - c.halfExtents;
      if (q.maxCoeff() > 0)
      {
        const Vec3 outside = q.cwiseMax(0.0);
        const double d = outside.norm();
        *n = outside.cwiseProduct(p.cwiseSign()) / std::max(d, 1e-12);
        return d;
      }
      int axis;
      const double d = q.maxCoeff(&axis);
      *n = Vec3::Zero();
      (*n)[axis] = p[axis] >= 0 ? 1.0 : -1.0;
      return d;
    }
    case ShapeType::Sphere:
    {
      const double length = p.norm();
      *n = length > 1e-12 ? Vec3(p / length) : Vec3::UnitZ();
      return length - c.radius;
    }
    case ShapeType::Capsule:
    {
      const Vec3 axis(0, 0, std::clamp(p.z(), -c.halfLength, c.halfLength));
      const Vec3 d = p - axis;
      const double length = d.norm();
      *n = length > 1e-12 ? Vec3(d / length) : Vec3::UnitX();
      return length - c.radius;
    }
    case ShapeType::Cylinder:
    {
      const double radial = std::hypot(p.x(), p.y());
      const double dr = radial - c.radius;
      const double dz = std::abs(p.z()) - c.halfLength;
      const Vec3 radialDir = radial > 1e-12 ? Vec3(p.x() / radial, p.y() / radial, 0)
                                            : Vec3::UnitX();
      const Vec3 axialDir(0, 0, p.z() >= 0 ? 1.0 : -1.0);
      if (dr > 0 && dz > 0)
      {
        const double d = std::hypot(dr, dz);
        *n = (radialDir * dr + axialDir * dz) / d;
        return d;
      }
      if (dr > dz)
      {
        *n = radialDir;
        return dr;
      }
      *n = axialDir;
      return dz;
    }
    case ShapeType::Plane:
      *n = Vec3::UnitZ();
      return p.z();
  }
  *n = Vec3::UnitZ();
  return 1e9;
}
}  // namespace

double ShapeDistance(const Collider &collider, const Vec3 &point, Vec3 *normal)
{
  Vec3 localNormal;
  const double d = LocalDistance(collider, collider.pose.inverse() * point, &localNormal);
  *normal = collider.pose.linear() * localNormal;
  return d;
}

Vec3 ClosestOnTriangle(const Vec3 &p, const Vec3 &a, const Vec3 &b, const Vec3 &c,
                       Vec3 *bary)
{
  // Ericson, Real-Time Collision Detection, 5.1.5.
  const Vec3 ab = b - a, ac = c - a, ap = p - a;
  const double d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) { *bary = {1, 0, 0}; return a; }
  const Vec3 bp = p - b;
  const double d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) { *bary = {0, 1, 0}; return b; }
  const double vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0)
  {
    const double v = d1 / (d1 - d3);
    *bary = {1 - v, v, 0};
    return a + v * ab;
  }
  const Vec3 cp = p - c;
  const double d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) { *bary = {0, 0, 1}; return c; }
  const double vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0)
  {
    const double w = d2 / (d2 - d6);
    *bary = {1 - w, 0, w};
    return a + w * ac;
  }
  const double va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
  {
    const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
    *bary = {0, 1 - w, w};
    return b + w * (c - b);
  }
  const double denom = 1.0 / (va + vb + vc);
  const double v = vb * denom, w = vc * denom;
  *bary = {1 - v - w, v, w};
  return a + ab * v + ac * w;
}

GarmentMesh MeshGarment(const GarmentPattern &pattern, double spacing)
{
  const PanelMesh panels[2] = {MeshPanel(pattern.front, spacing),
                               MeshPanel(pattern.back, spacing)};
  const PanelPattern *patterns[2] = {&pattern.front, &pattern.back};
  GarmentMesh mesh;
  std::vector<int64_t> particleOf[2];
  std::map<std::pair<int64_t, int64_t>, uint32_t> seamParticles;
  auto key = [](const Vec2 &p)
  { return std::make_pair(std::llround(p.x() * 1e6), std::llround(p.y() * 1e6)); };
  for (int panel = 0; panel < 2; ++panel)
  {
    const auto &pm = panels[panel];
    // Only vertices used by a triangle become particles.
    std::vector<bool> used(pm.points.size(), false);
    for (const auto &t : pm.triangles)
      for (auto v : t)
        used[v] = true;
    particleOf[panel].assign(pm.points.size(), -1);
    for (size_t i = 0; i < pm.points.size(); ++i)
    {
      if (!used[i])
        continue;
      if (pm.onSeam[i])
      {
        auto found = seamParticles.find(key(pm.points[i]));
        if (found != seamParticles.end())
        {
          particleOf[panel][i] = found->second;
          mesh.particlePanel[found->second] = 2;
          continue;
        }
      }
      particleOf[panel][i] = mesh.rest2d.size();
      if (pm.onSeam[i])
        seamParticles[key(pm.points[i])] = mesh.rest2d.size();
      mesh.rest2d.push_back(pm.points[i]);
      mesh.particlePanel.push_back(panel);
    }
  }
  mesh.mass.assign(mesh.rest2d.size(), 0.0);
  std::set<std::pair<uint32_t, uint32_t>> edges;
  double totalArea = 0;
  for (int panel = 0; panel < 2; ++panel)
  {
    const auto &pm = panels[panel];
    std::vector<uint32_t> renderOf(pm.points.size(), UINT32_MAX);
    std::map<std::pair<uint32_t, uint32_t>, std::vector<uint32_t>> opposite;
    for (const auto &t : pm.triangles)
    {
      std::array<uint32_t, 3> p{static_cast<uint32_t>(particleOf[panel][t[0]]),
                                static_cast<uint32_t>(particleOf[panel][t[1]]),
                                static_cast<uint32_t>(particleOf[panel][t[2]])};
      if (panel == 1)
        std::swap(p[1], p[2]);  // back panel faces -z
      const double area = std::abs(Cross2(mesh.rest2d[p[1]] - mesh.rest2d[p[0]],
                                          mesh.rest2d[p[2]] - mesh.rest2d[p[0]])) / 2;
      totalArea += area;
      if (panel == 0)
        mesh.frontArea += area;
      for (auto v : p)
        mesh.mass[v] += area / 3;
      mesh.triangles.push_back(p);
      mesh.trianglePanel.push_back(panel);
      std::array<uint32_t, 3> r;
      std::array<uint32_t, 3> source{t[0], panel == 1 ? t[2] : t[1], panel == 1 ? t[1] : t[2]};
      for (int k = 0; k < 3; ++k)
      {
        uint32_t &render = renderOf[source[k]];
        if (render == UINT32_MAX)
        {
          render = mesh.renderParticle.size();
          mesh.renderParticle.push_back(p[k]);
          const Vec2 &xy = pm.points[source[k]];
          mesh.renderUv.push_back(patterns[panel]->uvOffset +
                                  patterns[panel]->uvScale.cwiseProduct(xy));
        }
        r[k] = render;
      }
      mesh.renderTriangles.push_back(r);
      for (int k = 0; k < 3; ++k)
      {
        const uint32_t a = std::min(p[k], p[(k + 1) % 3]);
        const uint32_t b = std::max(p[k], p[(k + 1) % 3]);
        edges.insert({a, b});
        opposite[{a, b}].push_back(p[(k + 2) % 3]);
      }
    }
    for (const auto &[edge, others] : opposite)
    {
      if (others.size() != 2)
        continue;
      mesh.bends.push_back({others[0], others[1]});
      mesh.bendRest.push_back((mesh.rest2d[others[0]] - mesh.rest2d[others[1]]).norm());
    }
  }
  for (const auto &[a, b] : edges)
  {
    mesh.edges.push_back({a, b});
    mesh.edgeRest.push_back((mesh.rest2d[a] - mesh.rest2d[b]).norm());
  }
  for (auto &m : mesh.mass)
    m *= pattern.mass / totalArea;
  Vec2 lower = pattern.front.outline.front(), upper = lower;
  for (const auto &p : pattern.front.outline)
  {
    lower = lower.cwiseMin(p);
    upper = upper.cwiseMax(p);
  }
  mesh.extent = upper - lower;
  return mesh;
}

ClothWorld::ClothWorld(ClothParameters parameters) : parameters(parameters) {}

int ClothWorld::AddGarment(const std::string &id, const GarmentMesh &mesh, const Pose &pose)
{
  Garment garment;
  garment.id = id;
  garment.mesh = mesh;
  const size_t n = mesh.rest2d.size();
  garment.x.resize(n);
  for (size_t i = 0; i < n; ++i)
  {
    const double z = mesh.particlePanel[i] == 0 ? this->parameters.thickness / 2
                     : mesh.particlePanel[i] == 1 ? -this->parameters.thickness / 2 : 0.0;
    garment.x[i] = pose * Vec3(mesh.rest2d[i].x(), mesh.rest2d[i].y(), z);
  }
  garment.previous = garment.x;
  garment.v.assign(n, Vec3::Zero());
  garment.invMass.resize(n);
  for (size_t i = 0; i < n; ++i)
    garment.invMass[i] = 1.0 / mesh.mass[i];
  // Two-ring neighbourhoods are excluded from self collision: they are
  // connected through the mesh and their distance is kept by constraints.
  std::vector<std::vector<uint32_t>> adjacent(n);
  for (const auto &e : mesh.edges)
  {
    adjacent[e[0]].push_back(e[1]);
    adjacent[e[1]].push_back(e[0]);
  }
  garment.nearby.resize(n);
  for (size_t i = 0; i < n; ++i)
  {
    std::set<uint32_t> ring{static_cast<uint32_t>(i)};
    std::vector<uint32_t> frontier{static_cast<uint32_t>(i)};
    for (int depth = 0; depth < this->parameters.exclusionRings; ++depth)
    {
      std::vector<uint32_t> next;
      for (auto a : frontier)
        for (auto b : adjacent[a])
          if (ring.insert(b).second)
            next.push_back(b);
      frontier.swap(next);
    }
    garment.nearby[i].assign(ring.begin(), ring.end());
  }
  garment.bounds = Eigen::AlignedBox3d();
  for (const auto &p : garment.x)
    garment.bounds.extend(p);
  this->garments.push_back(std::move(garment));
  this->attachedMask.emplace_back(n, 0);
  return static_cast<int>(this->garments.size()) - 1;
}

void ClothWorld::SetPose(int garment, const std::vector<Vec3> &positions)
{
  auto &g = this->garments.at(garment);
  if (positions.size() != g.x.size())
    throw std::invalid_argument("Garment state size mismatch");
  g.x = positions;
  g.previous = positions;
  std::fill(g.v.begin(), g.v.end(), Vec3::Zero());
  g.bounds = Eigen::AlignedBox3d();
  for (const auto &p : g.x)
    g.bounds.extend(p);
  this->Wake(garment);
}

void ClothWorld::Wake(int garment)
{
  auto &g = this->garments.at(garment);
  if (g.asleep)
  {
    g.windowTime = 0;
    g.windowCentre = Vec3::Constant(1e9);  // the first window only sets a reference
  }
  g.asleep = false;
  g.calmTime = 0;
}

std::vector<bool> ClothWorld::AwakeMask() const
{
  std::vector<bool> mask;
  for (const auto &g : this->garments)
    mask.push_back(!g.asleep);
  return mask;
}

bool ClothWorld::Attached(int garment, uint32_t particle) const
{
  return this->attachedMask[garment][particle] != 0;
}

void ClothWorld::Predict(Garment &g, double h)
{
  const double damping = std::max(0.0, 1.0 - this->parameters.damping * h);
  for (size_t i = 0; i < g.x.size(); ++i)
  {
    g.previous[i] = g.x[i];
    g.v[i] = (g.v[i] + this->parameters.gravity * h) * damping;
    g.x[i] += g.v[i] * h;
  }
}

void ClothWorld::SolveDistances(Garment &g, double h)
{
  const int index = static_cast<int>(&g - this->garments.data());
  const auto &mask = this->attachedMask[index];
  auto solve = [&](uint32_t a, uint32_t b, double rest, double compliance)
  {
    const double wa = mask[a] ? 0.0 : g.invMass[a];
    const double wb = mask[b] ? 0.0 : g.invMass[b];
    const double w = wa + wb;
    if (w <= 0)
      return;
    Vec3 d = g.x[a] - g.x[b];
    const double length = d.norm();
    if (length < 1e-12)
      return;
    d /= length;
    const double c = length - rest;
    const double lambda = -c / (w + compliance / (h * h));
    g.x[a] += wa * lambda * d;
    g.x[b] -= wb * lambda * d;
  };
  const auto &mesh = g.mesh;
  for (size_t e = 0; e < mesh.edges.size(); ++e)
    solve(mesh.edges[e][0], mesh.edges[e][1], mesh.edgeRest[e],
          this->parameters.stretchCompliance);
  for (size_t b = 0; b < mesh.bends.size(); ++b)
    solve(mesh.bends[b][0], mesh.bends[b][1], mesh.bendRest[b],
          this->parameters.bendCompliance);
}

void ClothWorld::SolveShapes(Garment &g, double h, double fraction)
{
  const int index = static_cast<int>(&g - this->garments.data());
  const auto &mask = this->attachedMask[index];
  const double offset = this->parameters.contactOffset;
  const double step = 1.0 / this->parameters.substeps;
  const bool last = fraction > 1.0 - 1e-9;
  Eigen::AlignedBox3d region = g.bounds;
  region.min() -= Vec3::Constant(0.05);
  region.max() += Vec3::Constant(0.05);
  for (auto &collider : this->colliders)
  {
    if (!collider.bounds.intersects(region))
      continue;
    const Pose now = Interpolate(collider.previousPose, collider.pose, fraction);
    const Pose before = Interpolate(collider.previousPose, collider.pose, fraction - step);
    const Pose nowInverse = now.inverse();
    const Pose beforeInverse = before.inverse();
    const Pose carried = before * nowInverse;
    Eigen::AlignedBox3d box = collider.bounds;
    box.min() -= Vec3::Constant(offset);
    box.max() += Vec3::Constant(offset);
    int touching = 0;
    for (size_t i = 0; i < g.x.size(); ++i)
    {
      if (mask[i] || !box.contains(g.x[i]))
        continue;
      Vec3 localNormal;
      const double d = LocalDistance(collider, nowInverse * g.x[i], &localNormal);
      if (d >= offset)
        continue;
      ++touching;
      const Vec3 n = now.linear() * localNormal;
      const double depth = offset - d;
      const Vec3 start = g.x[i];
      g.x[i] += depth * n;
      Vec3 ignored;
      const double previousDistance = LocalDistance(collider, beforeInverse * g.previous[i], &ignored);
      g.previous[i] += std::clamp(offset - previousDistance, 0.0, depth) * n;
      // Coulomb friction against the surface motion at the contact point.
      const Vec3 surfaceMotion = g.x[i] - carried * g.x[i];
      Vec3 relative = (g.x[i] - g.previous[i]) - surfaceMotion;
      relative -= relative.dot(n) * n;
      const double slip = relative.norm();
      const double limit = collider.friction * depth;
      if (slip > 1e-12)
        g.x[i] -= relative * std::min(1.0, limit / slip);
      if (collider.body >= 0)
      {
        const Vec3 impulse = (g.x[i] - start) * (g.mesh.mass[i] / h);
        // Eigen vectors are not zeroed by default construction.
        auto &entry = this->bodyImpulses.try_emplace(collider.body, Vec3::Zero(), Vec3::Zero())
                          .first->second;
        entry.first -= impulse;
        g.bodyImpulse -= impulse;
        entry.second -= g.x[i].cross(impulse);
      }
    }
    if (last && touching > 0)
      g.contacts[collider.label] += touching;
  }
}

void ClothWorld::FindClothContacts(double /*dt*/)
{
  // Verlet list: pairs found within the thickness plus a skin stay valid
  // until some particle has moved half the skin since the search.
  const double skin = this->parameters.contactSkin;
  const double reach = this->parameters.thickness + skin;
  bool stale = this->searchAwake != this->AwakeMask() ||
               this->searchPositions.size() != this->garments.size();
  for (size_t a = 0; a < this->garments.size() && !stale; ++a)
  {
    const auto &g = this->garments[a];
    if (g.asleep)
      continue;
    const auto &before = this->searchPositions[a];
    if (before.size() != g.x.size())
    {
      stale = true;
      break;
    }
    for (size_t i = 0; i < g.x.size(); ++i)
      if ((g.x[i] - before[i]).squaredNorm() > 0.25 * skin * skin)
      {
        stale = true;
        break;
      }
  }
  if (!stale)
    return;
  this->searchAwake = this->AwakeMask();
  this->searchPositions.resize(this->garments.size());
  for (size_t a = 0; a < this->garments.size(); ++a)
    this->searchPositions[a] = this->garments[a].x;
  this->clothContacts.clear();
  if (!this->parameters.selfCollision)
    return;
  // Garments that can touch an awake garment.
  std::vector<int> active;
  for (size_t a = 0; a < this->garments.size(); ++a)
  {
    if (!this->garments[a].asleep)
    {
      active.push_back(a);
      continue;
    }
    Eigen::AlignedBox3d box = this->garments[a].bounds;
    box.min() -= Vec3::Constant(reach);
    box.max() += Vec3::Constant(reach);
    for (size_t b = 0; b < this->garments.size(); ++b)
      if (!this->garments[b].asleep && box.intersects(this->garments[b].bounds))
      {
        active.push_back(a);
        break;
      }
  }
  if (active.empty())
    return;
  // Particles of awake garments go into a spatial hash (one cell each, built
  // by counting sort into reused arrays); every triangle then visits the
  // cells under its inflated bounds.
  const double cell = 0.03;
  auto cellOf = [cell](double v) { return static_cast<int64_t>(std::floor(v / cell)); };
  auto hash = [](int64_t x, int64_t y, int64_t z)
  { return static_cast<uint64_t>((x * 73856093) ^ (y * 19349663) ^ (z * 83492791)); };
  auto &owners = this->hashSpans;  // garment and particle of each hashed point
  owners.clear();
  for (int a : active)
    if (!this->garments[a].asleep)
      for (uint32_t i = 0; i < this->garments[a].x.size(); ++i)
        owners.push_back({a, i, {}, {}});
  if (owners.empty())
    return;
  size_t buckets = 1024;
  while (buckets < 2 * owners.size())
    buckets *= 2;
  auto bucketOf = [&](const Vec3 &p)
  { return hash(cellOf(p.x()), cellOf(p.y()), cellOf(p.z())) & (buckets - 1); };
  auto &start = this->hashStart;
  auto &items = this->hashItems;
  auto &fill = this->hashFill;
  start.assign(buckets + 1, 0);
  items.resize(owners.size());
  for (const auto &o : owners)
    ++start[bucketOf(this->garments[o.garment].x[o.triangle]) + 1];
  for (size_t b = 0; b < buckets; ++b)
    start[b + 1] += start[b];
  fill.assign(start.begin(), start.end() - 1);
  // Positions are copied in bucket order so that the scan stays in cache.
  auto &points = this->hashPoints;
  points.resize(owners.size());
  for (uint32_t k = 0; k < owners.size(); ++k)
  {
    const Vec3 &p = this->garments[owners[k].garment].x[owners[k].triangle];
    const uint32_t slot = fill[bucketOf(p)]++;
    items[slot] = k;
    points[slot] = p;
  }
  // Nearest triangles per hashed particle, kept by insertion.
  const int keep = std::max(1, this->parameters.maxContacts);
  auto &best = this->hashBest;
  best.assign(owners.size() * keep, {std::numeric_limits<double>::max(), {}});
  for (int b : active)
  {
    const auto &other = this->garments[b];
    for (uint32_t t = 0; t < other.mesh.triangles.size(); ++t)
    {
      const auto &tri = other.mesh.triangles[t];
      const Vec3 &x0 = other.x[tri[0]], &x1 = other.x[tri[1]], &x2 = other.x[tri[2]];
      Eigen::AlignedBox3d box(x0);
      box.extend(x1);
      box.extend(x2);
      // Stretched or invalid triangles are skipped rather than flooding the grid.
      if (!box.sizes().allFinite() || box.sizes().maxCoeff() > 0.3)
        continue;
      const Vec3 centre = (x0 + x1 + x2) / 3;
      const double radius = std::sqrt(std::max({(x0 - centre).squaredNorm(),
                                                (x1 - centre).squaredNorm(),
                                                (x2 - centre).squaredNorm()})) + reach;
      const double radius2 = radius * radius;
      const Vec3 normal = (x1 - x0).cross(x2 - x0);
      for (int64_t x = cellOf(box.min().x() - reach); x <= cellOf(box.max().x() + reach); ++x)
        for (int64_t y = cellOf(box.min().y() - reach); y <= cellOf(box.max().y() + reach); ++y)
          for (int64_t z = cellOf(box.min().z() - reach); z <= cellOf(box.max().z() + reach); ++z)
          {
            const uint64_t bucket = hash(x, y, z) & (buckets - 1);
            for (uint32_t e = start[bucket]; e < start[bucket + 1]; ++e)
            {
              const Vec3 &p = points[e];
              if ((p - centre).squaredNorm() > radius2)
                continue;
              const uint32_t owner = items[e];
              const auto &o = owners[owner];
              const uint32_t i = o.triangle;
              if (o.garment == b)
              {
                const auto &near = this->garments[b].nearby[i];
                if (std::binary_search(near.begin(), near.end(), tri[0]) ||
                    std::binary_search(near.begin(), near.end(), tri[1]) ||
                    std::binary_search(near.begin(), near.end(), tri[2]))
                  continue;
              }
              Vec3 bary;
              const Vec3 c = ClosestOnTriangle(p, x0, x1, x2, &bary);
              const double distance = (p - c).norm();
              if (distance >= reach)
                continue;
              auto *slots = &best[owner * keep];
              if (distance >= slots[keep - 1].first)
                continue;
              bool duplicate = false;  // cells sharing a bucket repeat pairs
              for (int k = 0; k < keep && !duplicate; ++k)
                duplicate = slots[k].first < std::numeric_limits<double>::max() &&
                            slots[k].second.triangleGarment == b && slots[k].second.triangle == t;
              if (duplicate)
                continue;
              int k = keep - 1;
              while (k > 0 && slots[k - 1].first > distance)
              {
                slots[k] = slots[k - 1];
                --k;
              }
              const double side = (p - c).dot(normal) >= 0 ? 1.0 : -1.0;
              slots[k] = {distance, {o.garment, i, b, t, side}};
            }
          }
    }
  }
  for (const auto &slot : best)
    if (slot.first < std::numeric_limits<double>::max())
      this->clothContacts.push_back(slot.second);
  // Moving cloth disturbs the garments it touches.
  for (const auto &c : this->clothContacts)
  {
    auto &other = this->garments[c.triangleGarment];
    if (other.asleep && this->garments[c.garment].meanSpeed > 2 * this->parameters.sleepSpeed)
      this->Wake(c.triangleGarment);
  }
}

void ClothWorld::SolveClothContacts(double /*h*/)
{
  const double thickness = this->parameters.thickness;
  const double mu = this->parameters.clothFriction;
  for (const auto &c : this->clothContacts)
  {
    auto &g = this->garments[c.garment];
    auto &o = this->garments[c.triangleGarment];
    const auto &tri = o.mesh.triangles[c.triangle];
    const double wp = (g.asleep || this->attachedMask[c.garment][c.particle])
                          ? 0.0 : g.invMass[c.particle];
    double w[3];
    for (int k = 0; k < 3; ++k)
      w[k] = (o.asleep || this->attachedMask[c.triangleGarment][tri[k]]) ? 0.0
                                                                         : o.invMass[tri[k]];
    Vec3 bary;
    Vec3 &p = g.x[c.particle];
    const Vec3 closest = ClosestOnTriangle(p, o.x[tri[0]], o.x[tri[1]], o.x[tri[2]], &bary);
    Vec3 n = (o.x[tri[1]] - o.x[tri[0]]).cross(o.x[tri[2]] - o.x[tri[0]]);
    const double length = n.norm();
    if (length < 1e-14)
      continue;
    double distance, before;
    const Vec3 &q = g.previous[c.particle];
    const Vec3 closestBefore = bary[0] * o.previous[tri[0]] + bary[1] * o.previous[tri[1]] +
                               bary[2] * o.previous[tri[2]];
    if (bary.minCoeff() > 1e-3)
    {
      // Facing the triangle: keep the side it was on when the contact was found.
      n *= c.side / length;
      distance = (p - closest).dot(n);
      before = (q - closestBefore).dot(n);
    }
    else
    {
      // Beside an edge or corner the face normal says nothing about the gap.
      const Vec3 offset = p - closest;
      distance = offset.norm();
      if (distance < 1e-9)
        continue;
      n = offset / distance;
      before = (q - closestBefore).norm();
    }
    if (distance >= thickness)
      continue;
    const double denom = wp + bary[0] * bary[0] * w[0] + bary[1] * bary[1] * w[1] +
                         bary[2] * bary[2] * w[2];
    if (denom <= 0)
      continue;
    const double depth = (thickness - distance) * this->parameters.contactRelaxation;
    const double lambda = depth / denom;
    p += wp * lambda * n;
    for (int k = 0; k < 3; ++k)
      o.x[tri[k]] -= w[k] * bary[k] * lambda * n;
    // Overlap present before this substep is removed without adding speed:
    // squeezed layers otherwise turn every correction into jitter.
    const double stale =
        std::clamp((thickness - before) * this->parameters.contactRelaxation, 0.0, depth) / denom;
    g.previous[c.particle] += wp * stale * n;
    for (int k = 0; k < 3; ++k)
      o.previous[tri[k]] -= w[k] * bary[k] * stale * n;
    Vec3 relative = (p - g.previous[c.particle]);
    for (int k = 0; k < 3; ++k)
      relative -= bary[k] * (o.x[tri[k]] - o.previous[tri[k]]);
    relative -= relative.dot(n) * n;
    const double slip = relative.norm();
    if (slip < 1e-12)
      continue;
    const double scale = std::min(1.0, mu * depth / slip) / denom;
    p -= wp * scale * relative;
    for (int k = 0; k < 3; ++k)
      o.x[tri[k]] += w[k] * bary[k] * scale * relative;
    ++g.clothContacts;
  }
}

// Velocity pass after the position solve: touching layers do not bounce
// apart, which otherwise sets crumpled cloth vibrating and creeping.
void ClothWorld::DampClothContacts()
{
  const double reach = this->parameters.thickness * 1.05;
  for (const auto &c : this->clothContacts)
  {
    auto &g = this->garments[c.garment];
    auto &o = this->garments[c.triangleGarment];
    const auto &tri = o.mesh.triangles[c.triangle];
    const double wp = (g.asleep || this->attachedMask[c.garment][c.particle])
                          ? 0.0 : g.invMass[c.particle];
    double w[3];
    for (int k = 0; k < 3; ++k)
      w[k] = (o.asleep || this->attachedMask[c.triangleGarment][tri[k]]) ? 0.0
                                                                         : o.invMass[tri[k]];
    Vec3 bary;
    const Vec3 &p = g.x[c.particle];
    const Vec3 closest = ClosestOnTriangle(p, o.x[tri[0]], o.x[tri[1]], o.x[tri[2]], &bary);
    const Vec3 offset = p - closest;
    const double distance = offset.norm();
    if (distance >= reach || distance < 1e-9)
      continue;
    const Vec3 n = offset / distance;
    const double denom = wp + bary[0] * bary[0] * w[0] + bary[1] * bary[1] * w[1] +
                         bary[2] * bary[2] * w[2];
    if (denom <= 0)
      continue;
    Vec3 relative = g.v[c.particle];
    for (int k = 0; k < 3; ++k)
      relative -= bary[k] * o.v[tri[k]];
    const double normal = relative.dot(n);
    const double impulse = normal / denom;
    g.v[c.particle] -= wp * impulse * n;
    for (int k = 0; k < 3; ++k)
      o.v[tri[k]] += w[k] * bary[k] * impulse * n;
    // Fibres pressed together dissipate motion (internal friction of a pile).
    if (wp > 0)
      g.v[c.particle] *= 1.0 - this->parameters.contactDamping;
  }
}

void ClothWorld::ApplyAttachments(double fraction)
{
  for (const auto &a : this->attachments)
  {
    const auto &gripper = this->grippers[a.gripper];
    this->garments[a.garment].x[a.particle] =
        Interpolate(gripper.previousFingerA, gripper.fingerA, fraction) * a.local;
  }
}

void ClothWorld::UpdateSleep(double dt)
{
  std::vector<bool> held(this->garments.size(), false);
  for (const auto &a : this->attachments)
    held[a.garment] = true;
  for (size_t i = 0; i < this->garments.size(); ++i)
  {
    auto &g = this->garments[i];
    if (g.asleep)
      continue;
    // A few fidgeting particles must not keep a resting shirt awake.
    if (g.meanSpeed < this->parameters.sleepSpeed && g.maxSpeed < 10 * this->parameters.sleepSpeed &&
        !held[i])
      g.calmTime += dt;
    else
      g.calmTime = 0;
    // Squeezed piles keep fidgeting in place: a garment whose centre has not
    // moved over the sleep delay is at rest as well.
    Vec3 centre = Vec3::Zero();
    for (const auto &p : g.x)
      centre += p;
    centre /= static_cast<double>(std::max<size_t>(1, g.x.size()));
    g.windowTime += dt;
    bool still = false;
    if (g.windowTime >= this->parameters.sleepDelay)
    {
      still = !held[i] && (centre - g.windowCentre).norm() < 0.004 &&
              g.meanSpeed < 4 * this->parameters.sleepSpeed;
      g.windowCentre = centre;
      g.windowTime = 0;
    }
    if (g.calmTime >= this->parameters.sleepDelay || still)
    {
      g.asleep = true;
      std::fill(g.v.begin(), g.v.end(), Vec3::Zero());
    }
  }
}

void ClothWorld::Step(double dt)
{
  if (dt <= 0)
    return;
  const int substeps = std::max(1, this->parameters.substeps);
  const double h = dt / substeps;
  for (auto &c : this->colliders)
  {
    c.bounds = ColliderBounds(c);
    // Resting rigid bodies jitter by micrometres in the physics engine; only
    // real motion (above ~2 cm/s at the default step) disturbs sleeping cloth.
    c.moved = c.movable &&
              ((c.pose.translation() - c.previousPose.translation()).norm() > 1e-4 ||
               Eigen::AngleAxisd(c.pose.linear().transpose() * c.previousPose.linear()).angle() >
                   5e-4);
  }
  for (size_t i = 0; i < this->garments.size(); ++i)
  {
    auto &g = this->garments[i];
    g.contacts.clear();
    g.clothContacts = 0;
    g.bodyImpulse = Vec3::Zero();
    if (!g.asleep)
      continue;
    Eigen::AlignedBox3d region = g.bounds;
    region.min() -= Vec3::Constant(0.03);
    region.max() += Vec3::Constant(0.03);
    for (const auto &c : this->colliders)
      if (c.moved && c.bounds.intersects(region))
      {
        this->Wake(i);
        break;
      }
  }
  for (auto &mask : this->attachedMask)
    std::fill(mask.begin(), mask.end(), 0);
  for (const auto &a : this->attachments)
  {
    this->attachedMask[a.garment][a.particle] = 1;
    this->Wake(a.garment);
  }
  this->FindClothContacts(dt);
  for (auto &g : this->garments)
    if (!g.asleep)
      g.stepStart = g.x;
  for (int s = 0; s < substeps; ++s)
  {
    const double fraction = static_cast<double>(s + 1) / substeps;
    for (auto &g : this->garments)
      if (!g.asleep)
        this->Predict(g, h);
    this->ApplyAttachments(fraction);
    for (auto &g : this->garments)
      if (!g.asleep)
        this->SolveDistances(g, h);
    for (auto &g : this->garments)
      if (!g.asleep)
        this->SolveShapes(g, h, fraction);
    this->SolveClothContacts(h);
    this->ApplyAttachments(fraction);
    const double maxSpeed = this->parameters.maxSpeed;
    for (auto &g : this->garments)
    {
      if (g.asleep)
        continue;
      for (size_t i = 0; i < g.x.size(); ++i)
      {
        if (!g.x[i].allFinite())
          g.x[i] = g.previous[i];  // never let one bad contact poison the garment
        Vec3 v = (g.x[i] - g.previous[i]) / h;
        const double speed = v.norm();
        if (speed > maxSpeed)
          v *= maxSpeed / speed;
        g.v[i] = v;
      }
    }
    this->DampClothContacts();
  }
  for (auto &g : this->garments)
  {
    if (g.asleep)
      continue;
    g.bounds = Eigen::AlignedBox3d();
    double fastest = 0, total = 0;
    for (size_t i = 0; i < g.x.size(); ++i)
    {
      g.bounds.extend(g.x[i]);
      const double speed = (g.x[i] - g.stepStart[i]).norm() / dt;
      fastest = std::max(fastest, speed);
      total += speed;
    }
    g.maxSpeed = fastest;
    g.meanSpeed = total / std::max<size_t>(1, g.x.size());
  }
  this->UpdateSleep(dt);
  for (auto &c : this->colliders)
    c.previousPose = c.pose;
  for (auto &gripper : this->grippers)
    gripper.previousFingerA = gripper.fingerA;
}

int ClothWorld::UpdateGrasps(double closeGap, double openGap, double margin)
{
  int attached = 0;
  for (size_t index = 0; index < this->grippers.size(); ++index)
  {
    auto &gripper = this->grippers[index];
    const Vec3 centerA = gripper.fingerA * gripper.padA.center();
    const Vec3 centerB = gripper.fingerB * gripper.padB.center();
    Vec3 axis = centerB - centerA;
    const double distance = axis.norm();
    if (distance < 1e-9)
      continue;
    axis /= distance;
    auto halfAlong = [&](const Pose &pose, const Eigen::AlignedBox3d &pad)
    { return (pose.linear().transpose() * axis).cwiseAbs().dot(pad.sizes() / 2); };
    gripper.gap = distance - halfAlong(gripper.fingerA, gripper.padA) -
                  halfAlong(gripper.fingerB, gripper.padB);
    const bool closed = gripper.gap < (gripper.closed ? openGap : closeGap);
    if (closed && !gripper.closed)
    {
      // Pinch everything between the pads, across all garments.
      const Pose inverseA = gripper.fingerA.inverse();
      Eigen::AlignedBox3d pinch = gripper.padA;
      const Pose bInA = inverseA * gripper.fingerB;
      for (int corner = 0; corner < 8; ++corner)
        pinch.extend(bInA * gripper.padB.corner(
            static_cast<Eigen::AlignedBox3d::CornerType>(corner)));
      pinch.min() -= Vec3::Constant(margin);
      pinch.max() += Vec3::Constant(margin);
      for (size_t g = 0; g < this->garments.size(); ++g)
      {
        auto &garment = this->garments[g];
        for (uint32_t i = 0; i < garment.x.size(); ++i)
        {
          const Vec3 local = inverseA * garment.x[i];
          if (!pinch.contains(local) || this->Attached(g, i))
            continue;
          this->attachments.push_back({static_cast<int>(g), i, static_cast<int>(index), local});
          this->attachedMask[g][i] = 1;
          this->Wake(g);
          ++attached;
        }
      }
    }
    else if (!closed && gripper.closed)
    {
      this->ReleaseGripper(index);
    }
    gripper.closed = closed;
  }
  return attached;
}

void ClothWorld::ReleaseGripper(int gripper)
{
  for (const auto &a : this->attachments)
    if (a.gripper == gripper)
    {
      this->attachedMask[a.garment][a.particle] = 0;
      this->Wake(a.garment);
    }
  this->attachments.erase(
      std::remove_if(this->attachments.begin(), this->attachments.end(),
                     [gripper](const Attachment &a) { return a.gripper == gripper; }),
      this->attachments.end());
}

std::unordered_map<int, std::pair<Vec3, Vec3>> ClothWorld::TakeBodyImpulses()
{
  std::unordered_map<int, std::pair<Vec3, Vec3>> result;
  result.swap(this->bodyImpulses);
  return result;
}
// A tube of boxes approximating a cylindrical wall (axis along local z).
std::vector<Collider> Tube(const Pose &pose, double radius, double height, int sides,
                           const std::string &label)
{
  std::vector<Collider> result;
  const double thickness = 0.05;
  const double width = 2 * (radius + thickness) * std::tan(M_PI / sides) * 1.05;
  for (int k = 0; k < sides; ++k)
  {
    const double angle = 2 * M_PI * k / sides;
    Collider c;
    c.type = ShapeType::Box;
    c.label = label;
    c.halfExtents = Vec3(thickness / 2, width / 2, height / 2);
    Pose local = Pose::Identity();
    local.translation() = Vec3((radius + thickness / 2) * std::cos(angle),
                               (radius + thickness / 2) * std::sin(angle), height / 2);
    local.linear() = Eigen::AngleAxisd(angle, Vec3::UnitZ()).toRotationMatrix();
    c.pose = pose * local;
    c.previousPose = c.pose;
    c.friction = 0.4;
    result.push_back(c);
  }
  return result;
}


std::vector<Vec3> CrumpleGarment(const GarmentMesh &mesh, const ClothParameters &parameters,
                               uint32_t seed, double radius)
{
  ClothParameters p = parameters;
  p.sleepDelay = 0.3;
  ClothWorld w(p);
  Collider floor;
  floor.type = ShapeType::Plane;
  floor.label = "floor";
  floor.friction = 0.6;
  w.Colliders().push_back(floor);
  for (const auto &c : Tube(Pose::Identity(), radius, 1.2, 14, "tube"))
    w.Colliders().push_back(c);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  const size_t pick = std::uniform_int_distribution<size_t>(0, mesh.rest2d.size() - 1)(rng);
  Pose hang = Pose::Identity();
  hang.linear() = (Eigen::AngleAxisd(2 * M_PI * unit(rng), Vec3::UnitZ()) *
                   Eigen::AngleAxisd(M_PI / 2, Vec3::UnitX())).toRotationMatrix();
  hang.translation() = Vec3(0, 0, 1.0) - hang.linear() *
                       Vec3(mesh.rest2d[pick].x(), mesh.rest2d[pick].y(), 0);
  const int index = w.AddGarment("pile", mesh, hang);
  GripperState g;
  g.padA = Eigen::AlignedBox3d(Vec3(-0.01, -0.01, -0.02), Vec3(0.01, 0.01, -0.005));
  g.padB = Eigen::AlignedBox3d(Vec3(-0.01, -0.01, 0.005), Vec3(0.01, 0.01, 0.02));
  g.fingerA.translation() = w.Garments()[index].x[pick];
  g.previousFingerA = g.fingerA;
  g.fingerB = g.fingerA;
  g.fingerB.translation() += Vec3(0, 0.3, 0);
  w.Grippers().push_back(g);
  w.UpdateGrasps(0.012, 0.02, 0.0);
  w.Grippers()[0].fingerB = w.Grippers()[0].fingerA;
  w.UpdateGrasps(0.012, 0.02, 0.01);
  const double dt = 0.005;
  for (int i = 0; i < 60; ++i)
    w.Step(dt);
  // Lower the hanging shirt in a spiral, as when dropping laundry.
  const Vec3 top = w.Grippers()[0].fingerA.translation();
  const double turns = 2 + 3 * unit(rng);
  const double phase = 2 * M_PI * unit(rng);
  const int steps = 220;
  for (int i = 1; i <= steps; ++i)
  {
    const double t = static_cast<double>(i) / steps;
    const double a = phase + 2 * M_PI * turns * t;
    w.Grippers()[0].fingerA.translation() =
        Vec3(0.5 * radius * std::cos(a), 0.5 * radius * std::sin(a),
             top.z() + t * (0.06 - top.z()));
    w.Grippers()[0].fingerB = w.Grippers()[0].fingerA;
    w.Step(dt);
  }
  w.ReleaseGripper(0);
  for (int i = 0; i < 160 && !w.Garments()[index].asleep; ++i)
    w.Step(dt);
  auto x = w.Garments()[index].x;
  Vec3 center = Vec3::Zero();
  double bottom = 1e9;
  for (const auto &q : x)
  {
    center += q;
    bottom = std::min(bottom, q.z());
  }
  center /= static_cast<double>(x.size());
  for (auto &q : x)
    q -= Vec3(center.x(), center.y(), bottom);
  return x;
}

}  // namespace aha_laundry

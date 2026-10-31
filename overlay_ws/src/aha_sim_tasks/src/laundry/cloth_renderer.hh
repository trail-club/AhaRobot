// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#ifndef AHA_SIM_TASKS_LAUNDRY_CLOTH_RENDERER_HH_
#define AHA_SIM_TASKS_LAUNDRY_CLOTH_RENDERER_HH_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <gz/rendering/Scene.hh>

namespace aha_laundry
{
// Textured cloth surface drawn from simulated particle positions.
struct RenderMeshSpec
{
  std::string name;
  // One entry per render vertex. Seam particles appear once per panel so that
  // each panel keeps its own texture coordinates and shading normals.
  std::vector<uint32_t> particle;
  std::vector<float> uv;  // two floats per render vertex
  std::vector<uint32_t> indices;  // render-vertex triangles
  std::string albedoMap;
  std::string normalMap;
  float roughness = 0.9f;
};

// Updates deforming meshes in a gz-rendering ogre2 scene. Simulation threads
// call Publish(); the rendering thread calls Update() before each render.
class ClothRenderer
{
 public:
  ClothRenderer();
  ~ClothRenderer();

  void SetMeshes(std::vector<RenderMeshSpec> meshes);
  // Positions are xyz triplets of the garment's particles in world frame.
  void Publish(size_t mesh, const std::vector<float> &positions);

  // Rendering thread only. Creates resources on first use.
  void Update(const gz::rendering::ScenePtr &scene);
  // Rendering thread only. Releases Ogre resources before the scene goes away.
  void Destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}  // namespace aha_laundry

#endif

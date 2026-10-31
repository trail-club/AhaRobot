// Copyright 2026 trail-club
// SPDX-License-Identifier: Apache-2.0

#include "cloth_renderer.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <gz/common/Console.hh>
#include <gz/rendering/Material.hh>
#include <gz/rendering/ogre2/Ogre2Material.hh>
#include <gz/rendering/ogre2/Ogre2Scene.hh>

#include <OgreHlmsPbsDatablock.h>
#include <OgreItem.h>
#include <OgreMesh2.h>
#include <OgreMeshManager2.h>
#include <OgreRenderSystem.h>
#include <OgreRoot.h>
#include <OgreSceneManager.h>
#include <OgreSceneNode.h>
#include <OgreSubMesh2.h>
#include <Vao/OgreVaoManager.h>
#include <Vao/OgreVertexArrayObject.h>

namespace aha_laundry
{
namespace
{
constexpr size_t kFloatsPerVertex = 3 + 3 + 4 + 2;  // position, normal, tangent, uv

struct Vec3
{
  float x = 0, y = 0, z = 0;
};

Vec3 Sub(const float *a, const float *b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

Vec3 Cross(const Vec3 &a, const Vec3 &b)
{
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
}  // namespace

struct GpuMesh
{
  RenderMeshSpec spec;
  std::vector<float> positions;  // latest published particle positions
  bool dirty = false;
  bool hidden = true;
  // Rendering thread state.
  gz::rendering::MaterialPtr material;
  Ogre::MeshPtr mesh;
  Ogre::Item *item = nullptr;
  Ogre::SceneNode *node = nullptr;
  Ogre::VertexBufferPacked *vertexBuffer = nullptr;
  std::vector<float> vertexData;
};

struct ClothRenderer::Impl
{
  std::mutex mutex;
  std::vector<GpuMesh> meshes;
  std::vector<GpuMesh> retired;  // replaced meshes awaiting the rendering thread
  bool specsChanged = false;
  gz::rendering::ScenePtr scene;
  Ogre::SceneManager *sceneManager = nullptr;
  unsigned int generation = 0;

  void Release(GpuMesh &gpu)
  {
    if (gpu.item != nullptr && this->sceneManager != nullptr)
    {
      if (gpu.node != nullptr)
        gpu.node->detachObject(gpu.item);
      this->sceneManager->destroyItem(gpu.item);
    }
    gpu.item = nullptr;
    if (gpu.node != nullptr && this->sceneManager != nullptr)
      this->sceneManager->destroySceneNode(gpu.node);
    gpu.node = nullptr;
    if (gpu.mesh)
    {
      // Destroying the mesh releases its VAOs and buffers through the VaoManager.
      Ogre::MeshManager::getSingleton().remove(gpu.mesh);
      gpu.mesh.reset();
    }
    gpu.vertexBuffer = nullptr;
    if (gpu.material && this->scene)
      this->scene->DestroyMaterial(gpu.material);
    gpu.material.reset();
  }

  bool Create(GpuMesh &gpu)
  {
    const auto &spec = gpu.spec;
    const size_t vertices = spec.particle.size();
    if (vertices == 0 || spec.indices.empty() || spec.uv.size() != 2 * vertices)
    {
      gzerr << "[aha_laundry] Invalid render mesh " << spec.name << '\n';
      return false;
    }
    auto *root = Ogre::Root::getSingletonPtr();
    auto *vaoManager = root->getRenderSystem()->getVaoManager();
    Ogre::VertexElement2Vec elements;
    elements.push_back(Ogre::VertexElement2(Ogre::VET_FLOAT3, Ogre::VES_POSITION));
    elements.push_back(Ogre::VertexElement2(Ogre::VET_FLOAT3, Ogre::VES_NORMAL));
    elements.push_back(Ogre::VertexElement2(Ogre::VET_FLOAT4, Ogre::VES_TANGENT));
    elements.push_back(Ogre::VertexElement2(Ogre::VET_FLOAT2, Ogre::VES_TEXTURE_COORDINATES));
    gpu.vertexBuffer = vaoManager->createVertexBuffer(
        elements, vertices, Ogre::BT_DYNAMIC_PERSISTENT, nullptr, false);
    // Immutable buffers copy the initial data when keepAsShadow is false.
    std::vector<uint32_t> indices = spec.indices;
    auto *indexBuffer = vaoManager->createIndexBuffer(
        Ogre::IndexBufferPacked::IT_32BIT, indices.size(), Ogre::BT_IMMUTABLE,
        indices.data(), false);
    Ogre::VertexBufferPackedVec buffers{gpu.vertexBuffer};
    auto *vao = vaoManager->createVertexArrayObject(buffers, indexBuffer, Ogre::OT_TRIANGLE_LIST);
    const std::string name =
        "aha_laundry_" + spec.name + "_" + std::to_string(this->generation);
    gpu.mesh = Ogre::MeshManager::getSingleton().createManual(
        name, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
    auto *subMesh = gpu.mesh->createSubMesh();
    subMesh->mVao[Ogre::VpNormal].push_back(vao);
    subMesh->mVao[Ogre::VpShadow].push_back(vao);
    gpu.mesh->_setBounds(Ogre::Aabb(Ogre::Vector3::ZERO, Ogre::Vector3(50, 50, 50)), false);
    gpu.mesh->_setBoundingSphereRadius(90.0f);

    gpu.material = this->scene->CreateMaterial(name + "_material");
    gpu.material->SetDiffuse(1, 1, 1);
    gpu.material->SetAmbient(1, 1, 1);
    gpu.material->SetSpecular(0.05, 0.05, 0.05);
    gpu.material->SetRoughness(spec.roughness);
    gpu.material->SetMetalness(0.0f);
    if (!spec.albedoMap.empty())
      gpu.material->SetTexture(spec.albedoMap);
    if (!spec.normalMap.empty())
      gpu.material->SetNormalMap(spec.normalMap);
    auto ogreMaterial = std::dynamic_pointer_cast<gz::rendering::Ogre2Material>(gpu.material);
    if (!ogreMaterial)
    {
      gzerr << "[aha_laundry] Cloth rendering requires the ogre2 render engine\n";
      return false;
    }
    auto *datablock = ogreMaterial->Datablock();
    // Fabric is seen from both sides, e.g. the inside of a sleeve.
    datablock->setTwoSidedLighting(true, true);

    gpu.item = this->sceneManager->createItem(gpu.mesh, Ogre::SCENE_DYNAMIC);
    gpu.item->setDatablock(datablock);
    gpu.item->setCastShadows(true);
    gpu.node = this->sceneManager->getRootSceneNode(Ogre::SCENE_DYNAMIC)
        ->createChildSceneNode(Ogre::SCENE_DYNAMIC);
    gpu.node->attachObject(gpu.item);
    gpu.node->setVisible(!gpu.hidden);
    gpu.vertexData.assign(vertices * kFloatsPerVertex, 0.0f);
    for (size_t i = 0; i < vertices; ++i)
    {
      gpu.vertexData[i * kFloatsPerVertex + 10] = spec.uv[2 * i];
      gpu.vertexData[i * kFloatsPerVertex + 11] = spec.uv[2 * i + 1];
    }
    return true;
  }

  // Area-weighted normals and UV-aligned tangents, recomputed every update.
  static void FillVertices(GpuMesh &gpu, Ogre::Aabb &bounds)
  {
    const auto &spec = gpu.spec;
    const size_t vertices = spec.particle.size();
    std::vector<Vec3> normals(vertices), tangents(vertices), bitangents(vertices);
    float lower[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
    float upper[3] = {-lower[0], -lower[1], -lower[2]};
    auto position = [&](size_t v) { return &gpu.positions[3 * spec.particle[v]]; };
    for (size_t v = 0; v < vertices; ++v)
    {
      const float *p = position(v);
      float *out = &gpu.vertexData[v * kFloatsPerVertex];
      for (int k = 0; k < 3; ++k)
      {
        out[k] = p[k];
        lower[k] = std::min(lower[k], p[k]);
        upper[k] = std::max(upper[k], p[k]);
      }
    }
    for (size_t t = 0; t + 2 < spec.indices.size(); t += 3)
    {
      const uint32_t a = spec.indices[t], b = spec.indices[t + 1], c = spec.indices[t + 2];
      const Vec3 e1 = Sub(position(b), position(a));
      const Vec3 e2 = Sub(position(c), position(a));
      const Vec3 n = Cross(e1, e2);
      const float du1 = spec.uv[2 * b] - spec.uv[2 * a];
      const float dv1 = spec.uv[2 * b + 1] - spec.uv[2 * a + 1];
      const float du2 = spec.uv[2 * c] - spec.uv[2 * a];
      const float dv2 = spec.uv[2 * c + 1] - spec.uv[2 * a + 1];
      const float det = du1 * dv2 - du2 * dv1;
      const float r = std::abs(det) > 1e-12f ? 1.0f / det : 0.0f;
      const Vec3 tangent{(e1.x * dv2 - e2.x * dv1) * r, (e1.y * dv2 - e2.y * dv1) * r,
                         (e1.z * dv2 - e2.z * dv1) * r};
      const Vec3 bitangent{(e2.x * du1 - e1.x * du2) * r, (e2.y * du1 - e1.y * du2) * r,
                           (e2.z * du1 - e1.z * du2) * r};
      for (const uint32_t v : {a, b, c})
      {
        normals[v].x += n.x; normals[v].y += n.y; normals[v].z += n.z;
        tangents[v].x += tangent.x; tangents[v].y += tangent.y; tangents[v].z += tangent.z;
        bitangents[v].x += bitangent.x; bitangents[v].y += bitangent.y;
        bitangents[v].z += bitangent.z;
      }
    }
    for (size_t v = 0; v < vertices; ++v)
    {
      Vec3 n = normals[v];
      float length = std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
      n = length > 1e-12f ? Vec3{n.x / length, n.y / length, n.z / length} : Vec3{0, 0, 1};
      Vec3 t = tangents[v];
      const float dot = t.x * n.x + t.y * n.y + t.z * n.z;
      t = {t.x - n.x * dot, t.y - n.y * dot, t.z - n.z * dot};
      length = std::sqrt(t.x * t.x + t.y * t.y + t.z * t.z);
      t = length > 1e-12f ? Vec3{t.x / length, t.y / length, t.z / length} : Vec3{1, 0, 0};
      const Vec3 c = Cross(n, t);
      const Vec3 &b = bitangents[v];
      const float handedness = c.x * b.x + c.y * b.y + c.z * b.z < 0 ? -1.0f : 1.0f;
      float *out = &gpu.vertexData[v * kFloatsPerVertex];
      out[3] = n.x; out[4] = n.y; out[5] = n.z;
      out[6] = t.x; out[7] = t.y; out[8] = t.z; out[9] = handedness;
    }
    bounds = Ogre::Aabb::newFromExtents(Ogre::Vector3(lower[0], lower[1], lower[2]),
                                        Ogre::Vector3(upper[0], upper[1], upper[2]));
  }
};

ClothRenderer::ClothRenderer() : impl(std::make_unique<Impl>()) {}

ClothRenderer::~ClothRenderer() = default;

void ClothRenderer::SetMeshes(std::vector<RenderMeshSpec> meshes)
{
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  // Resources are released in the rendering thread on the next update.
  for (auto &old : this->impl->meshes)
    this->impl->retired.push_back(std::move(old));
  this->impl->meshes.clear();
  this->impl->meshes.resize(meshes.size());
  for (size_t i = 0; i < meshes.size(); ++i)
    this->impl->meshes[i].spec = std::move(meshes[i]);
  this->impl->specsChanged = true;
}

void ClothRenderer::Publish(size_t mesh, const std::vector<float> &positions)
{
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  if (mesh >= this->impl->meshes.size())
    return;
  auto &gpu = this->impl->meshes[mesh];
  gpu.positions = positions;
  gpu.dirty = true;
  gpu.hidden = false;
}

void ClothRenderer::Update(const gz::rendering::ScenePtr &scene)
{
  if (!scene)
    return;
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  if (this->impl->scene != scene)
  {
    auto ogreScene = std::dynamic_pointer_cast<gz::rendering::Ogre2Scene>(scene);
    if (!ogreScene)
    {
      gzerr << "[aha_laundry] Cloth rendering requires the ogre2 render engine\n";
      return;
    }
    this->impl->scene = scene;
    this->impl->sceneManager = ogreScene->OgreSceneManager();
    this->impl->specsChanged = true;
  }
  for (auto &gpu : this->impl->retired)
    this->impl->Release(gpu);
  this->impl->retired.clear();
  if (this->impl->specsChanged)
  {
    ++this->impl->generation;
    for (auto &gpu : this->impl->meshes)
    {
      this->impl->Release(gpu);
      if (!this->impl->Create(gpu))
        this->impl->Release(gpu);
      gpu.dirty = !gpu.positions.empty();
    }
    this->impl->specsChanged = false;
  }
  for (auto &gpu : this->impl->meshes)
  {
    if (!gpu.dirty || gpu.item == nullptr)
      continue;
    if (gpu.positions.size() < 3 * (*std::max_element(gpu.spec.particle.begin(),
                                                       gpu.spec.particle.end()) + 1))
      continue;
    Ogre::Aabb bounds;
    Impl::FillVertices(gpu, bounds);
    void *mapped = gpu.vertexBuffer->map(0, gpu.vertexBuffer->getNumElements());
    std::copy(gpu.vertexData.begin(), gpu.vertexData.end(), static_cast<float *>(mapped));
    gpu.vertexBuffer->unmap(Ogre::UO_KEEP_PERSISTENT);
    gpu.item->setLocalAabb(bounds);
    gpu.node->setVisible(!gpu.hidden);
    gpu.dirty = false;
  }
}

void ClothRenderer::Destroy()
{
  std::lock_guard<std::mutex> lock(this->impl->mutex);
  for (auto &gpu : this->impl->meshes)
    this->impl->Release(gpu);
  for (auto &gpu : this->impl->retired)
    this->impl->Release(gpu);
  this->impl->retired.clear();
  this->impl->scene.reset();
  this->impl->sceneManager = nullptr;
  this->impl->specsChanged = true;
}
}  // namespace aha_laundry

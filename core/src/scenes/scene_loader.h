#pragma once

#include "scene_builder.h"
#include "../misc/transform.cuh"

#pragma comment(lib, "Shlwapi.lib")
#define LIBCONFIGXX_STATIC
#include <libconfig/lib/libconfig.h++>

class scene_loader
{
public:
  scene_loader(std::string path);
  ~scene_loader() = default;

  scene_loader(const scene_loader &) = delete;

  scene_builder loadSceneFromFile();

private:
  std::string _path;
  libconfig::Config cfg;

  void loadPrimitives(scene_builder& builder, const libconfig::Setting& setting);
  void loadImageConfig(scene_builder& builder, const libconfig::Setting& setting);
  void loadImageBackgroundConfig(scene_builder& builder, const libconfig::Setting& setting);
  void loadCameraConfig(scene_builder& builder, const libconfig::Setting& setting);
  void loadTextures(scene_builder& builder, const libconfig::Setting& textures);
  void loadLights(scene_builder& builder, const libconfig::Setting& lights);
  void loadMaterials(scene_builder& builder, const libconfig::Setting& setting);
  void loadMeshes(scene_builder& builder, const libconfig::Setting& setting);
  void loadGroups(scene_builder& builder, const libconfig::Setting& setting);

  rt::transform applyTransform(const libconfig::Setting& primitive, scene_builder& builder, const char* name);
  point3 getPoint(const libconfig::Setting& setting);
  vector3 getVector(const libconfig::Setting& setting);
  color getColor(const libconfig::Setting& setting);
  uvmapping getUVmapping(const libconfig::Setting& setting);
  rt::transform getTransform(const libconfig::Setting& setting);


  void addSpherePrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addPlanePrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addQuadPrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addBoxPrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addConePrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addCylinderPrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addDiskPrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addTorusPrimitives(const libconfig::Setting& primitives, scene_builder& builder);
  void addVolumePrimitives(const libconfig::Setting& primitives, scene_builder& builder);


  void addImageTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addNormalTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addNoiseTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addSolidColorTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addCheckerTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addGradientColorTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addBumpTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addDisplacementTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addAlphaTexture(const libconfig::Setting& textures, scene_builder& builder);
  void addEmissiveTexture(const libconfig::Setting& textures, scene_builder& builder);


  void addLambertianMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addPhongMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addOrenNayarMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addIsotropicMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addAnisotropicMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addGlassMaterial(const libconfig::Setting& materials, scene_builder& builder);
  void addMetalMaterial(const libconfig::Setting& materials, scene_builder& builder);

  void addDirectionalLight(const libconfig::Setting& lights, scene_builder& builder);
  void addOmniLight(const libconfig::Setting& lights, scene_builder& builder);
  void addSpotLight(const libconfig::Setting& lights, scene_builder& builder);
};
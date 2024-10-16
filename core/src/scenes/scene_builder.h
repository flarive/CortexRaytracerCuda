#pragma once

#include "../primitives/hittable_list.cuh"
#include "../materials/material.cuh"
#include "../primitives/hittable.cuh"
#include "../misc/vector3.cuh"
#include "../misc/transform.cuh"
#include "../utilities/uvmapping.cuh"
#include "../misc/color.cuh"
#include "../textures/texture.cuh"
#include "../cameras/perspective_camera.cuh"
#include "scene_config.h"

//#include <map>





class scene_builder
{
    public:
        scene_builder();
        ~scene_builder() = default;

        [[nodiscard]] perspective_camera getCamera() const;
        //[[nodiscard]] hittable_list getSceneObjects() const;

        [[nodiscard]] imageConfig getImageConfig() const;
        [[nodiscard]] cameraConfig getCameraConfig() const;
        [[nodiscard]] lightsConfig getLightsConfig() const;
        [[nodiscard]] texturesConfig getTexturesConfig() const;
        [[nodiscard]] materialsConfig getMaterialsConfig() const;
        [[nodiscard]] primitivesConfig getPrimitivesConfig() const;
        [[nodiscard]] meshesConfig getMeshesConfig() const;

        // Image
        //scene_builder& setImageConfig(const imageConfig& config);
        scene_builder& setImageBackgroundConfig(const color& rgb, const char* filepath, bool is_skybox);
        scene_builder& imageSize(int width, int height);
        scene_builder& imageWidth(int width);
        scene_builder& imageHeight(int height);
        scene_builder& imageWidthWithAspectRatio(float aspectRatio);
        scene_builder& imageHeightWithAspectRatio(float aspectRatio);
        scene_builder& imageDepth(int depth);
        scene_builder& imageSamplesPerPixel(int samplesPerPixel);
        scene_builder& imageOutputFilePath(const char* filepath);

        // Camera
        //scene_builder& setCameraConfig(const cameraConfig& config);
        scene_builder& cameraAspectRatio(std::string aspectRatio);
        scene_builder& cameraOpeningTime(float time);
        scene_builder& cameraLookFrom(point3 point);
        scene_builder& cameraLookAt(point3 lookAt);
        scene_builder& cameraUpAxis(point3 vUp);
        scene_builder& cameraAperture(float aperture);
        scene_builder& cameraFocus(float focus);
        scene_builder& cameraFOV(float fov);
        scene_builder& cameraIsOrthographic(bool orthographic);
        scene_builder& cameraOrthoHeight(float height);

        // Textures
        scene_builder& initTexturesConfig(const uint32_t countSolidColor, const uint32_t countGradientColor, const uint32_t countImage, const uint32_t countPerlinNoise, const uint32_t countChecker, const uint32_t countBump, const uint32_t countNormal, const uint32_t countDisplacement, const uint32_t countAlpha, const uint32_t countEmissive);

        scene_builder& addSolidColorTexture(const char* textureName, color rgb);
        scene_builder& addCheckerTexture(const char* textureName, float scale, const char* oddTextureName, const char* evenTextureName, color oddColor, color evenColor);
        scene_builder& addNoiseTexture(const char* textureName, float scale = 1.0);
        scene_builder& addImageTexture(const char* textureName, const char* filepath);
        scene_builder& addNormalTexture(const char* textureName, const char* filepath, float strength);
        scene_builder& addGradientColorTexture(const char* textureName, color color1, color color2, bool aligned_v, bool hsv);
        scene_builder& addBumpTexture(const char* textureName, const char* filepath, float strength);
        scene_builder& addDisplacementTexture(const char* textureName, const char* filepath, float strength);
        scene_builder& addAlphaTexture(const char* textureName, const char* filepath, bool double_sided);
        scene_builder& addEmissiveTexture(const char* textureName, const char* filepath, float strength);

        // Materials
        scene_builder& initMaterialsConfig(const uint32_t countLambertian, const uint32_t countMetal, const uint32_t countDielectric, const uint32_t countIsotropic, const uint32_t countAnisotropic, const uint32_t countOrenNayar, const uint32_t countPhong);

        scene_builder& addGlassMaterial(const char* materialName, float refraction);
        scene_builder& addLambertianMaterial(const char* materialName, const color& rgb, const char* textureName);
        scene_builder& addPhongMaterial(const char* materialName, const char* diffuseTextureName, const char* specularTextureName, const char* normalTextureName, const char* bumpTextureName, const char* displaceTextureName, const char* alphaTextureName, const char* emissiveTextureName, const color& ambient, float shininess);
        scene_builder& addOrenNayarMaterial(const char* materialName, const color& rgb, const char* textureName, float albedo_temp, float roughness);
        scene_builder& addIsotropicMaterial(const char* materialName, const color& rgb, const char* textureName);
        scene_builder& addAnisotropicMaterial(const char* materialName, float nu, float nv, const color& rgb, const char* diffuseTextureName, const char* specularTextureName, const char* exponentTextureName);
        scene_builder& addMetalMaterial(const char* materialName, color rgb, float fuzz);
        

        // Lights
        scene_builder& initLightsConfig(const uint32_t countOmni, const uint32_t countDir, const uint32_t countSpot);

        scene_builder& addDirectionalLight(const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible, const char* name);
        scene_builder& addOmniDirectionalLight(const point3& pos, float radius, float intensity, color rgb, bool invisible, const char* name);
        scene_builder& addSpotLight(const point3& pos, const vector3& dir, float cosTotalWidth, float cosFalloffStart, float intensity, float radius, color rgb, bool invisible, const char* name);


        // Primitives
        scene_builder& initPrimitivesConfig(const uint32_t countSpherePrimitives, const uint32_t countPlanePrimitives, const uint32_t countQuadPrimitives, const uint32_t countBoxPrimitives, const uint32_t countConePrimitives, const uint32_t countCylinderPrimitives, const uint32_t countDiskPrimitives, const uint32_t countTorusPrimitives, const uint32_t countVolumePrimitives);

        scene_builder& addSphere(const char* name, point3 pos, float radius, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addQuad(const char* name, point3 position, vector3 u, vector3 v, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addPlane(const char* name, point3 p0, point3 p1, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addBox(const char* name, point3 p0, point3 p1, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addCylinder(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addCone(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addDisk(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addTorus(const char* name, point3 pos, float major_radius, float minor_radius, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs);
        scene_builder& addVolume(const char* name, const char* boundaryObjectName, float density, const color& rgb, const char* textureName, const char* groupName, const rt::transform& trs);

        // Meshes
        scene_builder& initMeshesConfig(const uint32_t countObjMeshes);

        scene_builder& addObjMesh(const char* name, point3 pos, const char* filepath, const char* materialName, bool use_mtl, bool use_smoothing, const char* groupName, const rt::transform& trs);

        // Groups
        scene_builder& addGroup(const char* name, bool& found);


    protected:
		imageConfig m_imageConfig{};
		cameraConfig m_cameraConfig{};
        lightsConfig m_lightsConfig{};
        texturesConfig m_texturesConfig{};
        materialsConfig m_materialsConfig{};
        primitivesConfig m_primitivesConfig{};
        meshesConfig m_meshesConfig{};
};
#pragma once

#include "../misc/vector3.cuh"
#include "../misc/color.cuh"



typedef struct {
    color rgb;
    const char* filepath;
    bool is_skybox;
} imageBackgroundConfig;

typedef struct {
    int height;
    int width;
    int depth;
    int spp;
    imageBackgroundConfig background;
    const char* outputFilePath;
} imageConfig;

typedef struct {
    float aspectRatio;
    float openingTime;
    point3 lookFrom;
    point3 lookAt;
    point3 upAxis;
    float aperture;
    float focus;
    bool isOrthographic;
    float fov; // for perspective cams
    float orthoHeight; // for orthographic cams
} cameraConfig;




typedef struct {
    point3 position;
    float radius;
    float intensity;
    color rgb;
    const char* name;
    bool invisible;
} omniLightConfig;

typedef struct {
    point3 position;
    vector3 u;
    vector3 v;
    float intensity;
    color rgb;
    const char* name;
    bool invisible;
} directionalLightConfig;

typedef struct {
    point3 position;
    vector3 direction;
    float cutoff;
    float falloff;
    float intensity;
    float radius;
    color rgb;
    const char* name;
    bool invisible;
} spotLightConfig;

typedef struct {
    omniLightConfig* omniLights;
    uint8_t omniLightCount;
    uint8_t omniLightCapacity;

    directionalLightConfig* dirLights;
    uint8_t dirLightCount;
    uint8_t dirLightCapacity;

    spotLightConfig* spotLights;
    uint8_t spotLightCount;
    uint8_t spotLightCapacity;
} lightsConfig;




typedef struct {
    const char* name;
    color rgb;
} solidColorTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
} imageTextureConfig;

typedef struct {
    const char* name;
    float scale;
} noiseTextureConfig;

typedef struct {
    const char* name;
    color color1;
    color color2;
    bool vertical;
    bool hsv;
} gradientColorTextureConfig;

typedef struct {
    const char* name;
    const char* oddTextureName;
    const char* evenTextureName;
    color oddColor;
    color evenColor;
    float scale;
} checkerTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
    float strength;
} bumpTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
    float strength;
} normalTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
    float strength;
} displacementTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
    bool doubleSided;
} alphaTextureConfig;

typedef struct {
    const char* name;
    const char* filepath;
    float strength;
} emissiveTextureConfig;

typedef struct {
    solidColorTextureConfig* solidColorTextures;
    uint8_t solidColorTextureCount;
    uint8_t solidColorTextureCapacity;

    gradientColorTextureConfig* gradientColorTextures;
    uint8_t gradientColorTextureCount;
    uint8_t gradientColorTextureCapacity;

    imageTextureConfig* imageTextures;
    uint8_t imageTextureCount;
    uint8_t imageTextureCapacity;

    checkerTextureConfig* checkerTextures;
    uint8_t checkerTextureCount;
    uint8_t checkerTextureCapacity;

    noiseTextureConfig* noiseTextures;
    uint8_t noiseTextureCount;
    uint8_t noiseTextureCapacity;

    bumpTextureConfig* bumpTextures;
    uint8_t bumpTextureCount;
    uint8_t bumpTextureCapacity;

    normalTextureConfig* normalTextures;
    uint8_t normalTextureCount;
    uint8_t normalTextureCapacity;

    displacementTextureConfig* displacementTextures;
    uint8_t displacementTextureCount;
    uint8_t displacementTextureCapacity;

    alphaTextureConfig* alphaTextures;
    uint8_t alphaTextureCount;
    uint8_t alphaTextureCapacity;

    emissiveTextureConfig* emissiveTextures;
    uint8_t emissiveTextureCount;
    uint8_t emissiveTextureCapacity;
} texturesConfig;






typedef struct {
    const char* name;
    const char* textureName;
    color rgb;
} lambertianMaterialConfig;

typedef struct {
    const char* name;
    color rgb;
    float fuzziness;
} metalMaterialConfig;

typedef struct {
    const char* name;
    float refraction;
} dielectricMaterialConfig;

typedef struct {
    const char* name;
    color rgb;
    const char* textureName;
} isotropicMaterialConfig;

typedef struct {
    const char* name;
    color rgb;
    float nuf;
    float nvf;
    const char* diffuseTextureName;
    const char* specularTextureName;
    const char* exponentTextureName;
    float roughness;
} anisotropicMaterialConfig;

typedef struct {
    const char* name;
    color rgb;
    const char* textureName;
    float albedo_temp;
    float roughness;
} orenNayarMaterialConfig;

typedef struct {
    const char* name;
    const char* diffuseTextureName;
    const char* specularTextureName;
    const char* bumpTextureName;
    const char* normalTextureName;
    const char* displacementTextureName;
    const char* alphaTextureName;
    const char* emissiveTextureName;
    color ambientColor;
    float shininess;
} phongMaterialConfig;




typedef struct {
    lambertianMaterialConfig* lambertianMaterials;
    uint8_t lambertianMaterialCount;
    uint8_t lambertianMaterialCapacity;

    metalMaterialConfig* metalMaterials;
    uint8_t metalMaterialCount;
    uint8_t metalMaterialCapacity;

    dielectricMaterialConfig* dielectricMaterials;
    uint8_t dielectricMaterialCount;
    uint8_t dielectricMaterialCapacity;

    isotropicMaterialConfig* isotropicMaterials;
    uint8_t isotropicMaterialCount;
    uint8_t isotropicMaterialCapacity;

    anisotropicMaterialConfig* anisotropicMaterials;
    uint8_t anisotropicMaterialCount;
    uint8_t anisotropicMaterialCapacity;

    orenNayarMaterialConfig* orenNayarMaterials;
    uint8_t orenNayarMaterialCount;
    uint8_t orenNayarMaterialCapacity;

    phongMaterialConfig* phongMaterials;
    uint8_t phongMaterialCount;
    uint8_t phongMaterialCapacity;
} materialsConfig;




typedef struct {
    imageConfig imageCfg;
    cameraConfig cameraCfg;
    lightsConfig lightsCfg;
    texturesConfig texturesCfg;
	materialsConfig materialsCfg;
} sceneConfig;



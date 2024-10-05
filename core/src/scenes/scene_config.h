#pragma once

#include "../misc/vector3.cuh"
#include "../misc/color.cuh"
#include "../misc/transform.cuh"
#include "../utilities/uvmapping.cuh"


struct imageBackgroundConfig  {
    color rgb;
    const char* filepath;
    bool is_skybox;
};

struct imageConfig {
    int height;
    int width;
    int depth;
    int spp;
    imageBackgroundConfig background;
    const char* outputFilePath;
};

struct cameraConfig {
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
};


//struct lightConfig{
//};

struct omniLightConfig {
    point3 position;
    float radius;
    float intensity;
    color rgb;
    const char* name;
    bool invisible;
};

struct directionalLightConfig {
    point3 position;
    vector3 u;
    vector3 v;
    float intensity;
    color rgb;
    const char* name;
    bool invisible;
};

struct spotLightConfig {
    point3 position;
    vector3 direction;
    float cutoff;
    float falloff;
    float intensity;
    float radius;
    color rgb;
    const char* name;
    bool invisible;
};



struct lightsConfig {
    omniLightConfig* omniLights;
    uint8_t omniLightCount;
    uint8_t omniLightCapacity;

    directionalLightConfig* dirLights;
    uint8_t dirLightCount;
    uint8_t dirLightCapacity;

    spotLightConfig* spotLights;
    uint8_t spotLightCount;
    uint8_t spotLightCapacity;
};






struct textureConfig {
    const char* name;
};

struct solidColorTextureConfig : public textureConfig {
    color rgb;
};

struct imageTextureConfig : public textureConfig {
    const char* filepath;
};

struct noiseTextureConfig : public textureConfig {
    float scale;
};

struct gradientColorTextureConfig : public textureConfig {
    color color1;
    color color2;
    bool vertical;
    bool hsv;
};

struct checkerTextureConfig : public textureConfig {
    const char* oddTextureName;
    const char* evenTextureName;
    color oddColor;
    color evenColor;
    float scale;
};

struct bumpTextureConfig : public textureConfig {
    const char* filepath;
    float strength;
};

struct normalTextureConfig : public textureConfig {
    const char* filepath;
    float strength;
};

struct displacementTextureConfig : public textureConfig {
    const char* filepath;
    float strength;
};

struct alphaTextureConfig : public textureConfig {
    const char* filepath;
    bool doubleSided;
};

struct emissiveTextureConfig : public textureConfig {
    const char* filepath;
    float strength;
};



struct texturesConfig {
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
};





struct materialConfig {
    const char* name;
};


struct lambertianMaterialConfig : public materialConfig {
    const char* textureName;
    color rgb;
};

struct metalMaterialConfig : public materialConfig {
    color rgb;
    float fuzziness;
};

struct dielectricMaterialConfig : public materialConfig {
    float refraction;
};

struct isotropicMaterialConfig : public materialConfig {
    color rgb;
    const char* textureName;
};

struct anisotropicMaterialConfig : public materialConfig {
    color rgb;
    float nuf;
    float nvf;
    const char* diffuseTextureName;
    const char* specularTextureName;
    const char* exponentTextureName;
    float roughness;
};

struct orenNayarMaterialConfig : public materialConfig {
    color rgb;
    const char* textureName;
    float albedo_temp;
    float roughness;
};

struct phongMaterialConfig : public materialConfig {
    const char* diffuseTextureName;
    const char* specularTextureName;
    const char* bumpTextureName;
    const char* normalTextureName;
    const char* displacementTextureName;
    const char* alphaTextureName;
    const char* emissiveTextureName;
    color ambientColor;
    float shininess;
};




struct materialsConfig {
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
};



//struct transformConfig
// {
//    vector3 translate;
//    vector3 rotate;
//    vector3 scale;
//};


//struct primitiveConfig {
//};

struct planePrimitiveConfig {
    const char* name;
    point3 point1;
    point3 point2;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};


struct boxPrimitiveConfig {
    const char* name;
    point3 position;
    vector3 size;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct quadPrimitiveConfig {
    const char* name;
    point3 position;
    vector3 u;
    vector3 v;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct spherePrimitiveConfig {
    const char* name;
    point3 position;
    float radius;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct cylinderPrimitiveConfig {
    const char* name;
    point3 position;
    float radius;
    float height;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct conePrimitiveConfig {
    const char* name;
    point3 position;
    float radius;
    float height;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct torusPrimitiveConfig {
    const char* name;
    point3 position;
    float major_radius;
    float minor_radius;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct diskPrimitiveConfig {
    const char* name;
    point3 position;
    float radius;
    float height;
    const char* materialName;
    uvmapping mapping;
    const char* groupName;
    rt::transform transform;
};

struct volumePrimitiveConfig {
    const char* name;
    const char* boundaryName;
    float density;
    color rgb;
    const char* textureName;
    const char* groupName;
    rt::transform transform;
};



struct primitivesConfig {
    spherePrimitiveConfig* spherePrimitives;
    uint8_t spherePrimitiveCount;
    uint8_t spherePrimitiveCapacity;

    boxPrimitiveConfig* boxPrimitives;
    uint8_t boxPrimitiveCount;
    uint8_t boxPrimitiveCapacity;

    quadPrimitiveConfig* quadPrimitives;
    uint8_t quadPrimitiveCount;
    uint8_t quadPrimitiveCapacity;

    planePrimitiveConfig* planePrimitives;
    uint8_t planePrimitiveCount;
    uint8_t planePrimitiveCapacity;

    cylinderPrimitiveConfig* cylinderPrimitives;
    uint8_t cylinderPrimitiveCount;
    uint8_t cylinderPrimitiveCapacity;

    conePrimitiveConfig* conePrimitives;
    uint8_t conePrimitiveCount;
    uint8_t conePrimitiveCapacity;

    torusPrimitiveConfig* torusPrimitives;
    uint8_t torusPrimitiveCount;
    uint8_t torusPrimitiveCapacity;

    diskPrimitiveConfig* diskPrimitives;
    uint8_t diskPrimitiveCount;
    uint8_t diskPrimitiveCapacity;

    volumePrimitiveConfig* volumePrimitives;
    uint8_t volumePrimitiveCount;
    uint8_t volumePrimitiveCapacity;
};



struct sceneConfig {
    imageConfig imageCfg;
    cameraConfig cameraCfg;
    lightsConfig lightsCfg;
    texturesConfig texturesCfg;
	materialsConfig materialsCfg;
    primitivesConfig primitivesCfg;
};
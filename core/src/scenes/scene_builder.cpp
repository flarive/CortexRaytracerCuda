#include "scene_builder.h"

#include "../primitives/box.cuh"
#include "../primitives/cone.cuh"
#include "../primitives/cylinder.cuh"
#include "../primitives/sphere.cuh"
#include "../primitives/aarect.cuh"

#include "../materials/dielectric.cuh"
#include "../materials/lambertian.cuh"
#include "../materials/phong.cuh"
#include "../materials/oren_nayar.cuh"
#include "../materials/diffuse_light.cuh"
#include "../materials/metal.cuh"
#include "../materials/isotropic.cuh"
#include "../materials/anisotropic.cuh"

#include "../textures/checker_texture.cuh"
#include "../textures/perlin_noise_texture.cuh"
#include "../textures/solid_color_texture.cuh"
#include "../textures/image_texture.cuh"
#include "../textures/normal_texture.cuh"
#include "../textures/gradient_texture.cuh"
#include "../textures/bump_texture.cuh"
#include "../textures/alpha_texture.cuh"
#include "../textures/displacement_texture.cuh"
#include "../textures/emissive_texture.cuh"

#include "../lights/directional_light.cuh"
#include "../lights/omni_light.cuh"

#include "../misc/bvh_node.cuh"

#include "../primitives/rotate.cuh"
#include "../primitives/translate.cuh"
#include "../primitives/scale.cuh"



#include <utility>

#include "../utilities/math_utils.cuh"
#include "scene_factory.h"

scene_builder::scene_builder()
{
  // Default image config
  this->m_imageConfig = { 225, 400, 100, 50, color(0.0f, 0.0f, 0.0f) };

  // Default camera config
  this->m_cameraConfig = { 16.0f / 9.0f, 0.0f, {0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 0.0f, 100.0f, false, 70.0f, 0.0f };
}

perspective_camera scene_builder::getCamera() const
{
    perspective_camera cam;
    cam.aspect_ratio = this->m_cameraConfig.aspectRatio;
    cam.background_color = color(0, 0, 0);
    cam.image_width = 512;
    cam.lookfrom = point3(this->m_cameraConfig.lookFrom.x, this->m_cameraConfig.lookFrom.y, this->m_cameraConfig.lookFrom.z);
    cam.lookat = point3(this->m_cameraConfig.lookAt.x, this->m_cameraConfig.lookAt.y, this->m_cameraConfig.lookAt.z);
    cam.vup = vector3(this->m_cameraConfig.upAxis.x, this->m_cameraConfig.upAxis.y, this->m_cameraConfig.upAxis.z);
    cam.vfov = this->m_cameraConfig.fov;
    cam.max_depth = 50;
    cam.samples_per_pixel = 100;
    cam.defocus_angle = this->m_cameraConfig.aperture;
    cam.focus_dist = this->m_cameraConfig.focus;

    // this->_camera.openingTime ???????????????

    return cam;
}

hittable_list scene_builder::getSceneObjects() const
{
  return this->m_objects;
}

imageConfig scene_builder::getImageConfig() const
{
  return this->m_imageConfig;
}

scene_builder& scene_builder::setImageBackgroundConfig(const color& rgb, const char* filepath, bool is_skybox)
{
    imageBackgroundConfig bgConfig;
    bgConfig.rgb = rgb;
    bgConfig.filepath = filepath;
    bgConfig.is_skybox = is_skybox;
    
    this->m_imageConfig.background = bgConfig;
	return *this;
}

scene_builder& scene_builder::imageSize(int width, int height)
{
  this->m_imageConfig.width = width;
  this->m_imageConfig.height = height;
  return *this;
}

scene_builder &scene_builder::imageWidth(int width)
{
  this->m_imageConfig.width = width;
  return *this;
}

scene_builder& scene_builder::imageHeight(int height)
{
  this->m_imageConfig.height = height;
  return *this;
}

scene_builder& scene_builder::imageWidthWithAspectRatio(float aspectRatio)
{
  this->m_imageConfig.width = int(float(this->m_imageConfig.height) * aspectRatio);
  return *this;
}

scene_builder& scene_builder::imageHeightWithAspectRatio(float aspectRatio)
{
  this->m_imageConfig.height = int(float(this->m_imageConfig.width) / aspectRatio);
  return *this;
}

scene_builder& scene_builder::imageDepth(int depth)
{
  this->m_imageConfig.depth = depth;
  return *this;
}

scene_builder& scene_builder::imageSamplesPerPixel(int samplesPerPixel)
{
  this->m_imageConfig.spp = samplesPerPixel;
  return *this;
}

scene_builder& scene_builder::imageOutputFilePath(const char* filepath)
{
    this->m_imageConfig.outputFilePath = filepath;
    return *this;
}

cameraConfig scene_builder::getCameraConfig() const
{
    return this->m_cameraConfig;
}

lightsConfig scene_builder::getLightsConfig() const
{
    return this->m_lightsConfig;
}

texturesConfig scene_builder::getTexturesConfig() const
{
    return this->m_texturesConfig;
}

scene_builder& scene_builder::cameraAspectRatio(std::string aspectRatio)
{
    float ratio = getRatio(aspectRatio.c_str());
    this->m_cameraConfig.aspectRatio = ratio;
  return *this;
}

scene_builder& scene_builder::cameraOpeningTime(float time)
{
  this->m_cameraConfig.openingTime = time;
  return *this;
}

scene_builder& scene_builder::cameraLookFrom(point3 point)
{
  this->m_cameraConfig.lookFrom = point;
  return *this;
}

scene_builder& scene_builder::cameraLookAt(point3 lookAt)
{
  this->m_cameraConfig.lookAt = lookAt;
  return *this;
}

scene_builder& scene_builder::cameraUpAxis(point3 vUp)
{
  this->m_cameraConfig.upAxis = vUp;
  return *this;
}

scene_builder& scene_builder::cameraAperture(float aperture)
{
  this->m_cameraConfig.aperture = aperture;
  return *this;
}

scene_builder& scene_builder::cameraFocus(float focus)
{
  this->m_cameraConfig.focus = focus;
  return *this;
}

scene_builder& scene_builder::cameraFOV(float fov)
{
  this->m_cameraConfig.fov = fov;
  return *this;
}

scene_builder& scene_builder::cameraIsOrthographic(bool orthographic)
{
    this->m_cameraConfig.isOrthographic = orthographic;
    return *this;
}

scene_builder& scene_builder::cameraOrthoHeight(float height)
{
    this->m_cameraConfig.orthoHeight = height;
    return *this;
}





scene_builder& scene_builder::initTexturesConfig(const uint32_t countSolidColor, const uint32_t countGradientColor, const uint32_t countImage, const uint32_t countPerlinNoise, const uint32_t countChecker, const uint32_t countBump, const uint32_t countNormal, const uint32_t countDisplacement, const uint32_t countAlpha, const uint32_t countEmissive)
{
    m_texturesConfig.solidColorTextureCount = 0;
    m_texturesConfig.solidColorTextureCapacity = countSolidColor;
    m_texturesConfig.solidColorTextures = new solidColorTextureConfig[countSolidColor];

    m_texturesConfig.gradientColorTextureCount = 0;
    m_texturesConfig.gradientColorTextureCapacity = countGradientColor;
    m_texturesConfig.gradientColorTextures = new gradientColorTextureConfig[countGradientColor];

    m_texturesConfig.imageTextureCount = 0;
    m_texturesConfig.imageTextureCapacity = countImage;
    m_texturesConfig.imageTextures = new imageTextureConfig[countImage];

    m_texturesConfig.noiseTextureCount = 0;
    m_texturesConfig.noiseTextureCapacity = countPerlinNoise;
    m_texturesConfig.noiseTextures = new noiseTextureConfig[countPerlinNoise];

    m_texturesConfig.checkerTextureCount = 0;
    m_texturesConfig.checkerTextureCapacity = countChecker;
    m_texturesConfig.checkerTextures = new checkerTextureConfig[countChecker];
    
    m_texturesConfig.bumpTextureCount = 0;
    m_texturesConfig.bumpTextureCapacity = countBump;
    m_texturesConfig.bumpTextures = new bumpTextureConfig[countBump];

    m_texturesConfig.normalTextureCount = 0;
    m_texturesConfig.normalTextureCapacity = countNormal;
    m_texturesConfig.normalTextures = new normalTextureConfig[countNormal];

    m_texturesConfig.displacementTextureCount = 0;
    m_texturesConfig.displacementTextureCapacity = countDisplacement;
    m_texturesConfig.displacementTextures = new displacementTextureConfig[countDisplacement];

    m_texturesConfig.alphaTextureCount = 0;
    m_texturesConfig.alphaTextureCapacity = countAlpha;
    m_texturesConfig.alphaTextures = new alphaTextureConfig[countAlpha];

    m_texturesConfig.emissiveTextureCount = 0;
    m_texturesConfig.emissiveTextureCapacity = countEmissive;
    m_texturesConfig.emissiveTextures = new emissiveTextureConfig[countEmissive];

    return *this;
}

scene_builder& scene_builder::addSolidColorTexture(const char* textureName, color rgb)
{
    //this->m_textures[textureName] = new solid_color_texture(rgb);

    // Get current count of solid color textures
    int c = m_texturesConfig.solidColorTextureCount;

    if (m_texturesConfig.solidColorTextureCount < m_texturesConfig.solidColorTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        m_texturesConfig.solidColorTextures[c] = solidColorTextureConfig{ textureName_copy, rgb };
        m_texturesConfig.solidColorTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of solid color textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addGradientColorTexture(const char* textureName, color color1, color color2, bool aligned_v, bool hsv)
{
	//this->m_textures[textureName] = new gradient_texture(color1, color2, aligned_v, hsv);

    // Get current count of gradient color textures
    int c = m_texturesConfig.gradientColorTextureCount;

    if (m_texturesConfig.gradientColorTextureCount < m_texturesConfig.gradientColorTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        m_texturesConfig.gradientColorTextures[c] = gradientColorTextureConfig{ textureName_copy, color1, color2, aligned_v, hsv };
        m_texturesConfig.gradientColorTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of gradient color textures." << std::endl;
    }

	return *this;
}

scene_builder& scene_builder::addCheckerTexture(const char* textureName, float scale, const char* oddTextureName, const char* evenTextureName, color oddColor, color evenColor)
{
	//this->m_textures[textureName] = new checker_texture(scale, oddColor, evenColor);

    // Get current count of checker textures
    int c = m_texturesConfig.checkerTextureCount;

    if (m_texturesConfig.checkerTextureCount < m_texturesConfig.checkerTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string


        char* oddTextureName_copy = nullptr;
        if (oddTextureName)
        {
            size_t length2 = strlen(oddTextureName) + 1;  // +1 for null terminator
            oddTextureName_copy = new char[length2]; // Allocate memory for the name
            strcpy(oddTextureName_copy, oddTextureName);  // Copy the string
        }

        char* evenTextureName_copy = nullptr;
        if (evenTextureName)
        {
            size_t length3 = strlen(evenTextureName) + 1;  // +1 for null terminator
            evenTextureName_copy = new char[length3]; // Allocate memory for the name
            strcpy(evenTextureName_copy, evenTextureName);  // Copy the string
        }

        m_texturesConfig.checkerTextures[c] = checkerTextureConfig{ textureName_copy, oddTextureName_copy, evenTextureName_copy, oddColor, evenColor, scale };
        m_texturesConfig.checkerTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of checker textures." << std::endl;
    }

	return *this;
}


scene_builder& scene_builder::addImageTexture(const char* textureName, const char* filepath)
{
    //this->m_textures[textureName] = new image_texture(img);


    // Get current count of gradient color textures
    int c = m_texturesConfig.imageTextureCount;

    if (m_texturesConfig.imageTextureCount < m_texturesConfig.imageTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.imageTextures[c] = imageTextureConfig{ textureName_copy, filepath_copy };
        m_texturesConfig.imageTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of image textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addNormalTexture(const char* textureName, const char* filepath, float strength)
{
    /*auto normal_tex = new image_texture(img);
    this->m_textures[textureName] = new normal_texture(normal_tex, strength);*/

    // Get current count of normal textures
    int c = m_texturesConfig.normalTextureCount;

    if (m_texturesConfig.normalTextureCount < m_texturesConfig.normalTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.normalTextures[c] = normalTextureConfig{ textureName_copy, filepath_copy, strength };
        m_texturesConfig.normalTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of normal textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addDisplacementTexture(const char* textureName, const char* filepath, float strength)
{
    //auto displace_tex = new image_texture(img);
    //this->m_textures[textureName] = new displacement_texture(displace_tex, strength);

    // Get current count of displacement textures
    int c = m_texturesConfig.displacementTextureCount;

    if (m_texturesConfig.displacementTextureCount < m_texturesConfig.displacementTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.displacementTextures[c] = displacementTextureConfig{ textureName_copy, filepath_copy, strength };
        m_texturesConfig.displacementTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of displacement textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addNoiseTexture(const char* textureName, float scale)
{
    //this->m_textures[textureName] = new perlin_noise_texture(scale);

    // Get current count of perlin noise textures
    int c = m_texturesConfig.noiseTextureCount;

    if (m_texturesConfig.noiseTextureCount < m_texturesConfig.noiseTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        m_texturesConfig.noiseTextures[c] = noiseTextureConfig{ textureName_copy, scale };
        m_texturesConfig.noiseTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of perlin noise textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addBumpTexture(const char* textureName, const char* filepath, float strength)
{
    /*auto bump_tex = new image_texture(img);
    this->m_textures[textureName] = new bump_texture(bump_tex, strength);*/

    // Get current count of bump textures
    int c = m_texturesConfig.bumpTextureCount;

    if (m_texturesConfig.bumpTextureCount < m_texturesConfig.bumpTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.bumpTextures[c] = bumpTextureConfig{ textureName_copy, filepath_copy, strength };
        m_texturesConfig.bumpTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of bump textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addAlphaTexture(const char* textureName, const char* filepath, bool double_sided)
{
    /*auto alpha_tex = new image_texture(img);
    this->m_textures[textureName] = new alpha_texture(alpha_tex, double_sided);*/

    // Get current count of alpha textures
    int c = m_texturesConfig.alphaTextureCount;

    if (m_texturesConfig.alphaTextureCount < m_texturesConfig.alphaTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.alphaTextures[c] = alphaTextureConfig{ textureName_copy, filepath_copy, double_sided };
        m_texturesConfig.alphaTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of alpha textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addEmissiveTexture(const char* textureName, const char* filepath, float strength)
{
    //auto emissive_tex = new image_texture(img);
    //this->m_textures[textureName] = new emissive_texture(emissive_tex, strength);

    // Get current count of emissive textures
    int c = m_texturesConfig.emissiveTextureCount;

    if (m_texturesConfig.emissiveTextureCount < m_texturesConfig.emissiveTextureCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length1]; // Allocate memory for the name
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length2 = strlen(filepath) + 1;  // +1 for null terminator
        char* filepath_copy = new char[length2]; // Allocate memory for the name
        strcpy(filepath_copy, filepath);  // Copy the string

        m_texturesConfig.emissiveTextures[c] = emissiveTextureConfig{ textureName_copy, filepath_copy, strength };
        m_texturesConfig.emissiveTextureCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of emissive textures." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addGlassMaterial(const char* materialName, float refraction)
{
  this->m_materials[materialName] = new dielectric(refraction);
  return *this;
}

scene_builder& scene_builder::addLambertianMaterial(const char* materialName, const color& rgb)
{
  this->m_materials[materialName] = new lambertian(rgb);
  return *this;
}

scene_builder& scene_builder::addLambertianMaterial(const char* materialName, const char* textureName)
{
  this->m_materials[materialName] = new lambertian(this->m_textures[textureName]);
  return *this;
}

scene_builder& scene_builder::addPhongMaterial(const char* materialName, const char* diffuseTextureName, const char* specularTextureName, const char* normalTextureName, const char* bumpTextureName, const char* displaceTextureName, const char* alphaTextureName, const char* emissiveTextureName, const color& ambient, float shininess)
{
    this->m_materials[materialName] = new phong(
        fetchTexture(diffuseTextureName),
        fetchTexture(specularTextureName),
        fetchTexture(bumpTextureName),
        fetchTexture(normalTextureName),
        fetchTexture(displaceTextureName),
        fetchTexture(alphaTextureName),
        fetchTexture(emissiveTextureName),
        ambient, shininess);
    return *this;
}

scene_builder& scene_builder::addOrenNayarMaterial(const char* materialName, const color& rgb, float albedo_temp, float roughness)
{
	this->m_materials[materialName] = new oren_nayar(rgb, albedo_temp, roughness);
	return *this;
}

scene_builder& scene_builder::addOrenNayarMaterial(const char* materialName, const char* textureName, float albedo_temp, float roughness)
{
	this->m_materials[materialName] = new oren_nayar(fetchTexture(textureName), albedo_temp, roughness);
	return *this;
}

scene_builder& scene_builder::addIsotropicMaterial(const char* materialName, const color& rgb)
{
    this->m_materials[materialName] = new isotropic(rgb);
    return *this;
}

scene_builder& scene_builder::addIsotropicMaterial(const char* materialName, const char* textureName)
{
    this->m_materials[materialName] = new isotropic(fetchTexture(textureName));
    return *this;
}

scene_builder& scene_builder::addAnisotropicMaterial(const char* materialName, float nu, float nv, const color& rgb)
{
    auto diffuse_tex = new solid_color_texture(rgb);
    this->m_materials[materialName] = new anisotropic(nu, nv, diffuse_tex, nullptr, nullptr);
    return *this;
}

scene_builder& scene_builder::addAnisotropicMaterial(const char* materialName, float nu, float nv, const char* diffuseTextureName, const char* specularTextureName, const char* exponentTextureName)
{
    this->m_materials[materialName] = new anisotropic(nu, nv, fetchTexture(diffuseTextureName), fetchTexture(specularTextureName), fetchTexture(exponentTextureName));
    return *this;
}

scene_builder& scene_builder::addMetalMaterial(const char* materialName, color rgb, float fuzz)
{
  this->m_materials[materialName] = new metal(rgb, fuzz);
  return *this;
}








scene_builder& scene_builder::initLightsConfig(const uint32_t countOmni, const uint32_t countDir, const uint32_t countSpot)
{
    m_lightsConfig.omniLightCount = 0;
    m_lightsConfig.omniLightCapacity = countOmni;
    m_lightsConfig.omniLights = new omniLightConfig[countOmni];

    m_lightsConfig.dirLightCount = 0;
    m_lightsConfig.dirLightCapacity = countDir;
    m_lightsConfig.dirLights = new directionalLightConfig[countDir];

    m_lightsConfig.spotLightCount = 0;
    m_lightsConfig.spotLightCapacity = countSpot;
    m_lightsConfig.spotLights = new spotLightConfig[countSpot];

    return *this;
}




scene_builder& scene_builder::addDirectionalLight(const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible, const char* name)
{
    /*this->m_objects.add(
        scene_factory::createDirectionalLight(
            name,
            pos,
            u,
            v,
            intensity,
            rgb,
            invisible
        )
    );*/

    // Get current count of directional lights
    int c = this->m_lightsConfig.dirLightCount;

    if (m_lightsConfig.dirLightCount < m_lightsConfig.dirLightCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        m_lightsConfig.dirLights[c] = directionalLightConfig{ pos, u, v, intensity, rgb, name_copy, invisible };
        m_lightsConfig.dirLightCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of directional lights." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addOmniDirectionalLight(const point3& pos, float radius, float intensity, color rgb, bool invisible, const char* name)
{
    //this->m_objects.add(
    //    scene_factory::createOmniDirectionalLight(
    //        name,
    //        pos,
    //        radius,
    //        intensity,
    //        rgb,
    //        invisible
    //    )
    //);


    // Get current count of omni lights
    int c = this->m_lightsConfig.omniLightCount;

    if (m_lightsConfig.omniLightCount < m_lightsConfig.omniLightCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        m_lightsConfig.omniLights[c] = omniLightConfig{ pos, radius, intensity, rgb, name_copy, invisible };
        m_lightsConfig.omniLightCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of omni lights." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addSpotLight(const point3& pos, const vector3& dir, float cutoff, float falloff, float intensity, float radius, color rgb, bool invisible, const char* name)
{
//    this->m_objects.add(
//        scene_factory::createSpotLight(
//            name,
//            pos,
//            dir,
//            cutoff,
//            falloff,
//            intensity,
//            radius,
//            rgb,
//            invisible
//        )
//    );

    // Get current count of spot lights
    int c = this->m_lightsConfig.spotLightCount;

    if (m_lightsConfig.spotLightCount < m_lightsConfig.spotLightCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        m_lightsConfig.spotLights[c] = spotLightConfig{ pos, dir, cutoff, falloff, intensity, radius, rgb, name_copy, invisible };
        m_lightsConfig.spotLightCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of spot lights." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addObject(hittable* obj)
{
  this->m_objects.add(obj);
  return *this;
}

scene_builder& scene_builder::addSphere(const char* name, point3 pos, float radius, const char* materialName, const uvmapping& uv, const char* group)
{
    auto sphere = scene_factory::createSphere(name, pos, radius, fetchMaterial(materialName), uv);

    if (group != nullptr && group[0] != '\0')
    {
        auto it = this->m_groups.find(group);
        if (it != this->m_groups.end())
        {
            // add to existing group is found
            hittable_list* grp = it->second;
            if (grp) { grp->add(sphere); }
        }
        else
        {
            // create group if not found
            this->m_groups.emplace(group, new hittable_list(sphere));
        }
    }
    else
    {
        this->m_objects.add(sphere);
    }

	return *this;
}

scene_builder& scene_builder::addPlane(const char* name, point3 p0, point3 p1, const char* materialName, const uvmapping& uv, const char* group)
{
    auto plane = scene_factory::createPlane(name, p0, p1, fetchMaterial(materialName), uv);
    
    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);
		if (it != this->m_groups.end())
		{
			// add to existing group is found
            hittable_list* grp = it->second;
			if (grp) { grp->add(plane); }
		}
		else
		{
			// create group if not found
			this->m_groups.emplace(group, new hittable_list(plane));
		}
	}
	else
	{
		this->m_objects.add(plane);
	}

    return *this;
}

scene_builder& scene_builder::addQuad(const char* name, point3 position, vector3 u, vector3 v, const char* materialName, const uvmapping& uv, const char* group)
{
    auto quad = scene_factory::createQuad(name, position, u, v, fetchMaterial(materialName), uv);
    
    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);
		if (it != this->m_groups.end())
		{
			// add to existing group is found
            hittable_list* grp = it->second;
			if (grp) { grp->add(quad); }
		}
		else
		{
			// create group if not found
			this->m_groups.emplace(group, new hittable_list(quad));
		}
	}
	else
	{
		this->m_objects.add(quad);
	}

    return *this;
}

scene_builder& scene_builder::addBox(const char* name, point3 p0, point3 p1, const char* materialName, const uvmapping& uv, const char* group)
{
    auto box = scene_factory::createBox(name, p0, p1, fetchMaterial(materialName), uv);

    if (group != nullptr && group[0] != '\0')
    {
        auto it = this->m_groups.find(group);

        if (it != this->m_groups.end())
        {
            // if key is found
            hittable_list* grp = it->second;
            if (grp)
            {
                grp->add(box);
            }
        }
        else
        {
            // if key is not found
            this->m_groups.emplace(group, new hittable_list(box));
        }
    }
    else
    {
        this->m_objects.add(box);
    }

    return *this;
}

scene_builder& scene_builder::addCylinder(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* group)
{
    auto cylinder = scene_factory::createCylinder(name, pos, radius, height, fetchMaterial(materialName), uv);
    
    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);

		if (it != this->m_groups.end())
		{
			// if key is found
            hittable_list* grp = it->second;
			if (grp)
			{
				grp->add(cylinder);
			}
		}
		else
		{
			// if key is not found
			this->m_groups.emplace(group, new hittable_list(cylinder));
		}
	}
	else
	{
		this->m_objects.add(cylinder);
	}
        
    return *this;
}

scene_builder& scene_builder::addDisk(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* group)
{
    auto disk = scene_factory::createDisk(name, pos, radius, height, fetchMaterial(materialName), uv);
    
    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);

		if (it != this->m_groups.end())
		{
			// if key is found
            hittable_list* grp = it->second;
			if (grp)
			{
				grp->add(disk);
			}
		}
		else
		{
			// if key is not found
			this->m_groups.emplace(group, new hittable_list(disk));
		}
	}
	else
	{
		this->m_objects.add(disk);
	}

    return *this;
}

scene_builder& scene_builder::addTorus(const char* name, point3 pos, float major_radius, float minor_radius, const char* materialName, const uvmapping& uv, const char* group)
{
    auto torus = scene_factory::createTorus(name, pos, major_radius, minor_radius, fetchMaterial(materialName), uv);

    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);

		if (it != this->m_groups.end())
		{
			// if key is found
            hittable_list* grp = it->second;
			if (grp)
			{
				grp->add(torus);
			}
		}
		else
		{
			// if key is not found
			this->m_groups.emplace(group, new hittable_list(torus));
		}
	}
	else
	{
		this->m_objects.add(torus);
	}
    
    return *this;
}

scene_builder& scene_builder::addCone(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* group)
{
    auto cone = scene_factory::createCone(name, pos, height, radius, fetchMaterial(materialName), uv);
    
    if (group != nullptr && group[0] != '\0')
	{
		auto it = this->m_groups.find(group);

		if (it != this->m_groups.end())
		{
			// if key is found
            hittable_list* grp = it->second;
			if (grp)
			{
				grp->add(cone);
			}
		}
		else
		{
			// if key is not found
			this->m_groups.emplace(group, new hittable_list(cone));
		}
	}
	else
	{
		this->m_objects.add(cone);
	}

    return *this;
}

scene_builder& scene_builder::addVolume(const char* name, const char* boundaryObjectName, float density, const char* textureName, const char* group)
{
    auto boundaryObject = this->m_objects.get(boundaryObjectName);
    if (boundaryObject)
    {
        auto volume = scene_factory::createVolume(name, boundaryObject, density, fetchTexture(textureName));

        if (group != nullptr && group[0] != '\0')
		{
			auto it = this->m_groups.find(group);

			if (it != this->m_groups.end())
			{
				// if key is found
                hittable_list* grp = it->second;
				if (grp)
				{
					grp->add(volume);
				}
			}
			else
			{
				// if key is not found
				this->m_groups.emplace(group, new hittable_list(volume));
			}
		}
		else
		{
			this->m_objects.add(volume);
		}

        this->m_objects.remove(boundaryObject);
    }

    return *this;
}

scene_builder& scene_builder::addVolume(const char* name, const char* boundaryObjectName, float density, const color& rgb, const char* group)
{
    auto boundaryObject = this->m_objects.get(boundaryObjectName);
    if (boundaryObject)
    {
        auto volume = scene_factory::createVolume(name, boundaryObject, density, rgb);

        if (group != nullptr && group[0] != '\0')
		{
			auto it = this->m_groups.find(group);

			if (it != this->m_groups.end())
			{
				// if key is found
                hittable_list* grp = it->second;
				if (grp)
				{
					grp->add(volume);
				}
			}
			else
			{
				// if key is not found
				this->m_groups.emplace(group, new hittable_list(volume));
			}
		}
		else
		{
			this->m_objects.add(volume);
		}

        this->m_objects.remove(boundaryObject);
    }

    return *this;
}

scene_builder& scene_builder::addMesh(const char* name, point3 pos, const char* filepath, const char* materialName, bool use_mtl, bool use_smoothing, const char* group)
{
    auto mesh = scene_factory::createMesh(name, pos, filepath, fetchMaterial(materialName), use_mtl, use_smoothing);

    if (group != nullptr && group[0] != '\0')
    {
        auto it = this->m_groups.find(group);

        if (it != this->m_groups.end())
        {
            // if key is found
            hittable_list* grp = it->second;
            if (grp)
            {
                grp->add(mesh);
            }
        }
        else
        {
            // if key is not found
            this->m_groups.emplace(group, new hittable_list(mesh));
        }
    }
    else
    {
        this->m_objects.add(mesh);
    }

	return *this;
}

scene_builder& scene_builder::addGroup(const char* name, bool& isUsed)
{
    isUsed = false;
    
    auto it = this->m_groups.find(name);

    if (it != this->m_groups.end())
    {
        hittable_list* group_objects = it->second;
        if (group_objects)
        {
            // ?????????????????????????
            int seed = 7896333;
            thrust::minstd_rand rng(seed);

            
            auto bvh_group = new bvh_node(group_objects->objects, 0, group_objects->object_count, rng, name);
            this->m_objects.add(bvh_group);

            isUsed = true;
        }
    }
    
    return *this;
}

scene_builder& scene_builder::translate(const vector3& vector, const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        hittable* found = this->m_objects.get(name);
        if (found)
        {
            found = new rt::translate(found, vector);
        }
        else
        {
            // search in groups
            for (auto& group : this->m_groups)
            {
                hittable* found2 = group.second->get(name);
                if (found2)
                {
                    found2 = new rt::translate(found2, vector);
                    break;
                }
            }
        }
    }
    else
    {
        hittable* back = this->m_objects.back();
        std::string n = back->getName();
        if (n == name)
        {
            hittable* tmp = this->m_objects.back();
            tmp = new rt::translate(back, vector);
        }
    }

    return *this;
}

scene_builder& scene_builder::rotate(const vector3& vector, const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        hittable* found = this->m_objects.get(name);
        if (found)
        {
            found = new rt::rotate(found, vector);
        }
        else
        {
            // search in groups
            for (auto& group : this->m_groups)
            {
                hittable* found2 = group.second->get(name);
                if (found2)
                {
                    found2 = new rt::rotate(found2, vector);
                    break;
                }
            }
        }
    }
    else
    {
        hittable* back = this->m_objects.back();
        char* n = back->getName();
        if (n == name)
        {
            hittable* tmp = this->m_objects.back();
            tmp = new rt::rotate(back, vector);
        }
    }

    return *this;
}

scene_builder& scene_builder::scale(const vector3& vector, const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        hittable* found = this->m_objects.get(name);
        if (found)
        {
            found = new  rt::scale(found, vector);
        }
        else
        {
            // search in groups
            for (auto& group : this->m_groups)
            {
                hittable* found2 = group.second->get(name);
                if (found2)
                {
                    found2 = new rt::scale(found2, vector);
                    break;
                }
            }
        }
    }
    else
    {
        hittable* back = this->m_objects.back();
        std::string n = back->getName();
        if (n == name)
        {
            hittable* tmp = this->m_objects.back();
            tmp= new rt::scale(back, vector);
        }
    }

    return *this;
}

material* scene_builder::fetchMaterial(const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        auto it = this->m_materials.find(name);

        if (it != this->m_materials.end())
        {
            // if key is found
            return it->second;
        }
        else
        {
            // if key is not found
            std::cerr << "[WARN] Material " << name << " not found !" << std::endl;
            return nullptr;
        }
    }

    return nullptr;
}

texture* scene_builder::fetchTexture(const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        auto it = this->m_textures.find(name);

        if (it != this->m_textures.end())
        {
            // if key is found
            return it->second;
        }
        else
        {
            // if key is not found
            std::cerr << "[WARN] Texture " << name << " not found !" << std::endl;
            return nullptr;
        }
    }

    return nullptr;
}
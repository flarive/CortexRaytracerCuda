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

//hittable_list scene_builder::getSceneObjects() const
//{
//  return this->m_objects;
//}

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

materialsConfig scene_builder::getMaterialsConfig() const
{
    return this->m_materialsConfig;
}

primitivesConfig scene_builder::getPrimitivesConfig() const
{
    return this->m_primitivesConfig;
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




scene_builder& scene_builder::initMaterialsConfig(const uint32_t countLambertian, const uint32_t countMetal, const uint32_t countDielectric, const uint32_t countIsotropic, const uint32_t countAnisotropic, const uint32_t countOrenNayar, const uint32_t countPhong)
{
    m_materialsConfig.lambertianMaterialCount = 0;
    m_materialsConfig.lambertianMaterialCapacity = countLambertian;
    m_materialsConfig.lambertianMaterials = new lambertianMaterialConfig[countLambertian];

    m_materialsConfig.metalMaterialCount = 0;
    m_materialsConfig.metalMaterialCapacity = countMetal;
    m_materialsConfig.metalMaterials = new metalMaterialConfig[countMetal];

    m_materialsConfig.dielectricMaterialCount = 0;
    m_materialsConfig.dielectricMaterialCapacity = countDielectric;
    m_materialsConfig.dielectricMaterials = new dielectricMaterialConfig[countDielectric];

    m_materialsConfig.isotropicMaterialCount = 0;
    m_materialsConfig.isotropicMaterialCapacity = countIsotropic;
    m_materialsConfig.isotropicMaterials = new isotropicMaterialConfig[countIsotropic];

    m_materialsConfig.anisotropicMaterialCount = 0;
    m_materialsConfig.anisotropicMaterialCapacity = countAnisotropic;
    m_materialsConfig.anisotropicMaterials = new anisotropicMaterialConfig[countAnisotropic];

    m_materialsConfig.orenNayarMaterialCount = 0;
    m_materialsConfig.orenNayarMaterialCapacity = countOrenNayar;
    m_materialsConfig.orenNayarMaterials = new orenNayarMaterialConfig[countOrenNayar];

    m_materialsConfig.phongMaterialCount = 0;
    m_materialsConfig.phongMaterialCapacity = countPhong;
    m_materialsConfig.phongMaterials = new phongMaterialConfig[countPhong];

    return *this;
}





scene_builder& scene_builder::addLambertianMaterial(const char* materialName, const color& rgb, const char* textureName)
{
    //this->m_materials[materialName] = new lambertian(rgb);

    // Get current count of lambertian materials
    int c = m_materialsConfig.lambertianMaterialCount;

    if (c < m_materialsConfig.lambertianMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length1]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        // When assigning the textureName, allocate memory and copy the string
        size_t length2 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length2]; // Allocate memory for the textureName
        strcpy(textureName_copy, textureName);  // Copy the string

        m_materialsConfig.lambertianMaterials[c] = lambertianMaterialConfig{ materialName_copy, textureName_copy, rgb };
        m_materialsConfig.lambertianMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of lambertian materials." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addMetalMaterial(const char* materialName, color rgb, float fuzz)
{
    //this->m_materials[materialName] = new metal(rgb, fuzz);

    // Get current count of metal materials
    int c = m_materialsConfig.metalMaterialCount;

    if (c < m_materialsConfig.metalMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        m_materialsConfig.metalMaterials[c] = metalMaterialConfig{ materialName_copy, rgb, fuzz };
        m_materialsConfig.metalMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of metal materials." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addGlassMaterial(const char* materialName, float refraction)
{
    //this->m_materials[materialName] = new dielectric(refraction);

    // Get current count of glass materials
    int c = m_materialsConfig.dielectricMaterialCount;

    if (c < m_materialsConfig.dielectricMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        m_materialsConfig.dielectricMaterials[c] = dielectricMaterialConfig{ materialName_copy, refraction };
        m_materialsConfig.dielectricMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of glass materials." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addPhongMaterial(const char* materialName, const char* diffuseTextureName, const char* specularTextureName, const char* normalTextureName, const char* bumpTextureName, const char* displaceTextureName, const char* alphaTextureName, const char* emissiveTextureName, const color& ambient, float shininess)
{
    /*this->m_materials[materialName] = new phong(
        fetchTexture(diffuseTextureName),
        fetchTexture(specularTextureName),
        fetchTexture(bumpTextureName),
        fetchTexture(normalTextureName),
        fetchTexture(displaceTextureName),
        fetchTexture(alphaTextureName),
        fetchTexture(emissiveTextureName),
        ambient, shininess);*/

    // Get current count of phong materials
    int c = m_materialsConfig.phongMaterialCount;

    if (c < m_materialsConfig.phongMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length1]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length2 = strlen(diffuseTextureName) + 1;  // +1 for null terminator
        char* diffuseTextureName_copy = new char[length2]; // Allocate memory
        strcpy(diffuseTextureName_copy, diffuseTextureName);  // Copy the string

        size_t length3 = strlen(specularTextureName) + 1;  // +1 for null terminator
        char* specularTextureName_copy = new char[length3]; // Allocate memory
        strcpy(specularTextureName_copy, specularTextureName);  // Copy the string

        size_t length4 = strlen(normalTextureName) + 1;  // +1 for null terminator
        char* normalTextureName_copy = new char[length4]; // Allocate memory
        strcpy(normalTextureName_copy, normalTextureName);  // Copy the string

        size_t length5 = strlen(bumpTextureName) + 1;  // +1 for null terminator
        char* bumpTextureName_copy = new char[length5]; // Allocate memory
        strcpy(bumpTextureName_copy, bumpTextureName);  // Copy the string

        size_t length6 = strlen(displaceTextureName) + 1;  // +1 for null terminator
        char* displaceTextureName_copy = new char[length6]; // Allocate memory
        strcpy(displaceTextureName_copy, displaceTextureName);  // Copy the string

        size_t length7 = strlen(alphaTextureName) + 1;  // +1 for null terminator
        char* alphaTextureName_copy = new char[length7]; // Allocate memory
        strcpy(alphaTextureName_copy, alphaTextureName);  // Copy the string

        size_t length8 = strlen(emissiveTextureName) + 1;  // +1 for null terminator
        char* emissiveTextureName_copy = new char[length8]; // Allocate memory
        strcpy(emissiveTextureName_copy, emissiveTextureName);  // Copy the string

        m_materialsConfig.phongMaterials[c] = phongMaterialConfig{ materialName_copy, diffuseTextureName_copy, specularTextureName_copy, normalTextureName_copy, bumpTextureName_copy, displaceTextureName_copy, alphaTextureName_copy, emissiveTextureName_copy, ambient, shininess };
        m_materialsConfig.phongMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of phong materials." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addOrenNayarMaterial(const char* materialName, const color& rgb, const char* textureName, float albedo_temp, float roughness)
{
	//this->m_materials[materialName] = new oren_nayar(rgb, albedo_temp, roughness);

    // Get current count of oren nayar materials
    int c = m_materialsConfig.orenNayarMaterialCount;

    if (c < m_materialsConfig.orenNayarMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length1]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length2 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length2]; // Allocate memory
        strcpy(textureName_copy, textureName);  // Copy the string

        m_materialsConfig.orenNayarMaterials[c] = orenNayarMaterialConfig{ materialName_copy, rgb, textureName_copy, albedo_temp, roughness };
        m_materialsConfig.orenNayarMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of oren nayar materials." << std::endl;
    }

	return *this;
}



scene_builder& scene_builder::addIsotropicMaterial(const char* materialName, const color& rgb, const char* textureName)
{
    //this->m_materials[materialName] = new isotropic(rgb);

    // Get current count of isotropic materials
    int c = m_materialsConfig.isotropicMaterialCount;

    if (c < m_materialsConfig.isotropicMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length1]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length2 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length2]; // Allocate memory
        strcpy(textureName_copy, textureName);  // Copy the string

        m_materialsConfig.isotropicMaterials[c] = isotropicMaterialConfig{ materialName_copy, rgb, textureName_copy };
        m_materialsConfig.isotropicMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of isotropic materials." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addAnisotropicMaterial(const char* materialName, float nu, float nv, const color& rgb, const char* diffuseTextureName, const char* specularTextureName, const char* exponentTextureName)
{
    /*auto diffuse_tex = new solid_color_texture(rgb);
    this->m_materials[materialName] = new anisotropic(nu, nv, diffuse_tex, nullptr, nullptr);*/

    // Get current count of anisotropic materials
    int c = m_materialsConfig.anisotropicMaterialCount;

    if (c < m_materialsConfig.anisotropicMaterialCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length1]; // Allocate memory for the name
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length2 = strlen(diffuseTextureName) + 1;  // +1 for null terminator
        char* diffuseTextureName_copy = new char[length2]; // Allocate memory
        strcpy(diffuseTextureName_copy, diffuseTextureName);  // Copy the string

        size_t length3 = strlen(specularTextureName) + 1;  // +1 for null terminator
        char* specularTextureName_copy = new char[length3]; // Allocate memory
        strcpy(specularTextureName_copy, specularTextureName);  // Copy the string

        size_t length4 = strlen(exponentTextureName) + 1;  // +1 for null terminator
        char* exponentTextureName_copy = new char[length4]; // Allocate memory
        strcpy(exponentTextureName_copy, exponentTextureName);  // Copy the string

        m_materialsConfig.anisotropicMaterials[c] = anisotropicMaterialConfig{ materialName_copy, rgb, nu, nv, diffuseTextureName_copy, specularTextureName_copy, exponentTextureName_copy };
        m_materialsConfig.anisotropicMaterialCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of anisotropic materials." << std::endl;
    }

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

//scene_builder& scene_builder::addObject(hittable* obj)
//{
//  this->m_objects.add(obj);
//  return *this;
//}


scene_builder& scene_builder::initPrimitivesConfig(const uint32_t countSpherePrimitives, const uint32_t countPlanePrimitives, const uint32_t countQuadPrimitives, const uint32_t countBoxPrimitives, const uint32_t countConePrimitives, const uint32_t countCylinderPrimitives, const uint32_t countDiskPrimitives, const uint32_t countTorusPrimitives, const uint32_t countVolumePrimitives)
{
    m_primitivesConfig.spherePrimitiveCount = 0;
    m_primitivesConfig.spherePrimitiveCapacity = countSpherePrimitives;
    m_primitivesConfig.spherePrimitives = new spherePrimitiveConfig[countSpherePrimitives];

    m_primitivesConfig.planePrimitiveCount = 0;
    m_primitivesConfig.planePrimitiveCapacity = countPlanePrimitives;
    m_primitivesConfig.planePrimitives = new planePrimitiveConfig[countPlanePrimitives];

    m_primitivesConfig.quadPrimitiveCount = 0;
    m_primitivesConfig.quadPrimitiveCapacity = countQuadPrimitives;
    m_primitivesConfig.quadPrimitives = new quadPrimitiveConfig[countQuadPrimitives];

    m_primitivesConfig.boxPrimitiveCount = 0;
    m_primitivesConfig.boxPrimitiveCapacity = countBoxPrimitives;
    m_primitivesConfig.boxPrimitives = new boxPrimitiveConfig[countBoxPrimitives];

    m_primitivesConfig.conePrimitiveCount = 0;
    m_primitivesConfig.conePrimitiveCapacity = countConePrimitives;
    m_primitivesConfig.conePrimitives = new conePrimitiveConfig[countConePrimitives];

    m_primitivesConfig.cylinderPrimitiveCount = 0;
    m_primitivesConfig.cylinderPrimitiveCapacity = countCylinderPrimitives;
    m_primitivesConfig.cylinderPrimitives = new cylinderPrimitiveConfig[countCylinderPrimitives];

    m_primitivesConfig.diskPrimitiveCount = 0;
    m_primitivesConfig.diskPrimitiveCapacity = countDiskPrimitives;
    m_primitivesConfig.diskPrimitives = new diskPrimitiveConfig[countDiskPrimitives];

    m_primitivesConfig.torusPrimitiveCount = 0;
    m_primitivesConfig.torusPrimitiveCapacity = countTorusPrimitives;
    m_primitivesConfig.torusPrimitives = new torusPrimitiveConfig[countTorusPrimitives];

    m_primitivesConfig.volumePrimitiveCount = 0;
    m_primitivesConfig.volumePrimitiveCapacity = countVolumePrimitives;
    m_primitivesConfig.volumePrimitives = new volumePrimitiveConfig[countVolumePrimitives];

    return *this;
}

scene_builder& scene_builder::addSphere(const char* name, point3 pos, float radius, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
    //auto sphere = scene_factory::createSphere(name, pos, radius, fetchMaterial(materialName), uv);

    //if (groupName != nullptr && groupName[0] != '\0')
    //{
    //    auto it = this->m_groups.find(groupName);
    //    if (it != this->m_groups.end())
    //    {
    //        // add to existing group is found
    //        hittable_list* grp = it->second;
    //        if (grp) { grp->add(sphere); }
    //    }
    //    else
    //    {
    //        // create group if not found
    //        this->m_groups.emplace(groupName, new hittable_list(sphere));
    //    }
    //}
    //else
    //{
    //    this->m_objects.add(sphere);
    //}

    // Get current count of sphere primitives
    int c = this->m_primitivesConfig.spherePrimitiveCount;

    if (m_primitivesConfig.spherePrimitiveCount < m_primitivesConfig.spherePrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.spherePrimitives[c] = spherePrimitiveConfig{ name_copy, pos, radius, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.spherePrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of sphere primitives." << std::endl;
    }

	return *this;
}

scene_builder& scene_builder::addPlane(const char* name, point3 p0, point3 p1, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto plane = scene_factory::createPlane(name, p0, p1, fetchMaterial(materialName), uv);
 //   
 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);
	//	if (it != this->m_groups.end())
	//	{
	//		// add to existing group is found
 //           hittable_list* grp = it->second;
	//		if (grp) { grp->add(plane); }
	//	}
	//	else
	//	{
	//		// create group if not found
	//		this->m_groups.emplace(groupName, new hittable_list(plane));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(plane);
	//}

    // Get current count of plane primitives
    int c = this->m_primitivesConfig.planePrimitiveCount;

    if (m_primitivesConfig.planePrimitiveCount < m_primitivesConfig.planePrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.planePrimitives[c] = planePrimitiveConfig{ name_copy, p0, p1, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.planePrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of planes primitives." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addQuad(const char* name, point3 position, vector3 u, vector3 v, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto quad = scene_factory::createQuad(name, position, u, v, fetchMaterial(materialName), uv);
 //   
 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);
	//	if (it != this->m_groups.end())
	//	{
	//		// add to existing group is found
 //           hittable_list* grp = it->second;
	//		if (grp) { grp->add(quad); }
	//	}
	//	else
	//	{
	//		// create group if not found
	//		this->m_groups.emplace(groupName, new hittable_list(quad));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(quad);
	//}

    // Get current count of quad primitives
    int c = this->m_primitivesConfig.quadPrimitiveCount;

    if (m_primitivesConfig.quadPrimitiveCount < m_primitivesConfig.quadPrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.quadPrimitives[c] = quadPrimitiveConfig{ name_copy, position, u, v, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.quadPrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of quad primitives." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addBox(const char* name, point3 position, point3 size, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
    //auto box = scene_factory::createBox(name, p0, p1, fetchMaterial(materialName), uv);

    //if (groupName != nullptr && groupName[0] != '\0')
    //{
    //    auto it = this->m_groups.find(groupName);

    //    if (it != this->m_groups.end())
    //    {
    //        // if key is found
    //        hittable_list* grp = it->second;
    //        if (grp)
    //        {
    //            grp->add(box);
    //        }
    //    }
    //    else
    //    {
    //        // if key is not found
    //        this->m_groups.emplace(groupName, new hittable_list(box));
    //    }
    //}
    //else
    //{
    //    this->m_objects.add(box);
    //}

    // Get current count of box primitives
    int c = this->m_primitivesConfig.boxPrimitiveCount;

    if (m_primitivesConfig.boxPrimitiveCount < m_primitivesConfig.boxPrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        auto box = boxPrimitiveConfig{ name_copy, position, size, materialName_copy, uv, groupName_copy, trs };

        m_primitivesConfig.boxPrimitives[c] = box;
        m_primitivesConfig.boxPrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of box primitives." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addCylinder(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto cylinder = scene_factory::createCylinder(name, pos, radius, height, fetchMaterial(materialName), uv);
 //   
 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);

	//	if (it != this->m_groups.end())
	//	{
	//		// if key is found
 //           hittable_list* grp = it->second;
	//		if (grp)
	//		{
	//			grp->add(cylinder);
	//		}
	//	}
	//	else
	//	{
	//		// if key is not found
	//		this->m_groups.emplace(groupName, new hittable_list(cylinder));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(cylinder);
	//}

    // Get current count of cylinder primitives
    int c = this->m_primitivesConfig.cylinderPrimitiveCount;

    if (m_primitivesConfig.cylinderPrimitiveCount < m_primitivesConfig.cylinderPrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.cylinderPrimitives[c] = cylinderPrimitiveConfig{ name_copy, pos, radius, height, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.cylinderPrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of cylinder primitives." << std::endl;
    }
        
    return *this;
}

scene_builder& scene_builder::addDisk(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto disk = scene_factory::createDisk(name, pos, radius, height, fetchMaterial(materialName), uv);
 //   
 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);

	//	if (it != this->m_groups.end())
	//	{
	//		// if key is found
 //           hittable_list* grp = it->second;
	//		if (grp)
	//		{
	//			grp->add(disk);
	//		}
	//	}
	//	else
	//	{
	//		// if key is not found
	//		this->m_groups.emplace(groupName, new hittable_list(disk));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(disk);
	//}

    // Get current count of disk primitives
    int c = this->m_primitivesConfig.diskPrimitiveCount;

    if (m_primitivesConfig.diskPrimitiveCount < m_primitivesConfig.diskPrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.diskPrimitives[c] = diskPrimitiveConfig{ name_copy, pos, radius, height, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.diskPrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of disk primitives." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addTorus(const char* name, point3 pos, float major_radius, float minor_radius, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto torus = scene_factory::createTorus(name, pos, major_radius, minor_radius, fetchMaterial(materialName), uv);

 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);

	//	if (it != this->m_groups.end())
	//	{
	//		// if key is found
 //           hittable_list* grp = it->second;
	//		if (grp)
	//		{
	//			grp->add(torus);
	//		}
	//	}
	//	else
	//	{
	//		// if key is not found
	//		this->m_groups.emplace(groupName, new hittable_list(torus));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(torus);
	//}

    // Get current count of torus primitives
    int c = this->m_primitivesConfig.torusPrimitiveCount;

    if (m_primitivesConfig.torusPrimitiveCount < m_primitivesConfig.torusPrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.torusPrimitives[c] = torusPrimitiveConfig{ name_copy, pos, major_radius, minor_radius, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.torusPrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of torus primitives." << std::endl;
    }
    
    return *this;
}

scene_builder& scene_builder::addCone(const char* name, point3 pos, float radius, float height, const char* materialName, const uvmapping& uv, const char* groupName, const rt::transform& trs)
{
 //   auto cone = scene_factory::createCone(name, pos, height, radius, fetchMaterial(materialName), uv);
 //   
 //   if (groupName != nullptr && groupName[0] != '\0')
	//{
	//	auto it = this->m_groups.find(groupName);

	//	if (it != this->m_groups.end())
	//	{
	//		// if key is found
 //           hittable_list* grp = it->second;
	//		if (grp)
	//		{
	//			grp->add(cone);
	//		}
	//	}
	//	else
	//	{
	//		// if key is not found
	//		this->m_groups.emplace(groupName, new hittable_list(cone));
	//	}
	//}
	//else
	//{
	//	this->m_objects.add(cone);
	//}

    // Get current count of cone primitives
    int c = this->m_primitivesConfig.conePrimitiveCount;

    if (m_primitivesConfig.conePrimitiveCount < m_primitivesConfig.conePrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(materialName) + 1;  // +1 for null terminator
        char* materialName_copy = new char[length2]; // Allocate memory
        strcpy(materialName_copy, materialName);  // Copy the string

        size_t length3 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length3]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.conePrimitives[c] = conePrimitiveConfig{ name_copy, pos, radius, height, materialName_copy, uv, groupName_copy, trs };
        m_primitivesConfig.planePrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of cone primitives." << std::endl;
    }

    return *this;
}

scene_builder& scene_builder::addVolume(const char* name, const char* boundaryObjectName, float density, const color& rgb, const char* textureName, const char* groupName, const rt::transform& trs)
{
  //  auto boundaryObject = this->m_objects.get(boundaryObjectName);
  //  if (boundaryObject)
  //  {
  //      auto volume = scene_factory::createVolume(name, boundaryObject, density, fetchTexture(textureName));

  //      if (groupName != nullptr && groupName[0] != '\0')
		//{
		//	auto it = this->m_groups.find(groupName);

		//	if (it != this->m_groups.end())
		//	{
		//		// if key is found
  //              hittable_list* grp = it->second;
		//		if (grp)
		//		{
		//			grp->add(volume);
		//		}
		//	}
		//	else
		//	{
		//		// if key is not found
		//		this->m_groups.emplace(groupName, new hittable_list(volume));
		//	}
		//}
		//else
		//{
		//	this->m_objects.add(volume);
		//}

  //      this->m_objects.remove(boundaryObject);
  //  }

    // Get current count of volume primitives
    int c = this->m_primitivesConfig.volumePrimitiveCount;

    if (m_primitivesConfig.volumePrimitiveCount < m_primitivesConfig.volumePrimitiveCapacity)
    {
        // When assigning the name, allocate memory and copy the string
        size_t length1 = strlen(name) + 1;  // +1 for null terminator
        char* name_copy = new char[length1]; // Allocate memory for the name
        strcpy(name_copy, name);  // Copy the string

        size_t length2 = strlen(boundaryObjectName) + 1;  // +1 for null terminator
        char* boundaryObjectName_copy = new char[length2]; // Allocate memory
        strcpy(boundaryObjectName_copy, boundaryObjectName);  // Copy the string

        size_t length3 = strlen(textureName) + 1;  // +1 for null terminator
        char* textureName_copy = new char[length3]; // Allocate memory
        strcpy(textureName_copy, textureName);  // Copy the string

        size_t length4 = strlen(groupName) + 1;  // +1 for null terminator
        char* groupName_copy = new char[length4]; // Allocate memory
        strcpy(groupName_copy, groupName);  // Copy the string

        m_primitivesConfig.volumePrimitives[c] = volumePrimitiveConfig{ name_copy, boundaryObjectName_copy, density, rgb, textureName_copy, groupName_copy, trs };
        m_primitivesConfig.volumePrimitiveCount++;
    }
    else {
        // Handle error, for example, log a message or throw an exception
        std::cerr << "Exceeded maximum number of volume primitives." << std::endl;
    }

    return *this;
}


scene_builder& scene_builder::addMesh(const char* name, point3 pos, const char* filepath, const char* materialName, bool use_mtl, bool use_smoothing, const char* groupName)
{
    //auto mesh = scene_factory::createMesh(name, pos, filepath, fetchMaterial(materialName), use_mtl, use_smoothing);

    //if (groupName != nullptr && groupName[0] != '\0')
    //{
    //    auto it = this->m_groups.find(groupName);

    //    if (it != this->m_groups.end())
    //    {
    //        // if key is found
    //        hittable_list* grp = it->second;
    //        if (grp)
    //        {
    //            grp->add(mesh);
    //        }
    //    }
    //    else
    //    {
    //        // if key is not found
    //        this->m_groups.emplace(groupName, new hittable_list(mesh));
    //    }
    //}
    //else
    //{
    //    this->m_objects.add(mesh);
    //}

	return *this;
}

scene_builder& scene_builder::addGroup(const char* name, bool& isUsed)
{
    isUsed = false;
    
    //auto it = this->m_groups.find(name);

    //if (it != this->m_groups.end())
    //{
    //    hittable_list* group_objects = it->second;
    //    if (group_objects)
    //    {
    //        // ?????????????????????????
    //        int seed = 7896333;
    //        thrust::minstd_rand rng(seed);

    //        
    //        auto bvh_group = new bvh_node(group_objects->objects, 0, group_objects->object_count, rng, name);
    //        this->m_objects.add(bvh_group);

    //        isUsed = true;
    //    }
    //}
    
    return *this;
}

scene_builder& scene_builder::translate(const vector3& vector, const char* name)
{
    //if (name != nullptr && name[0] != '\0')
    //{
    //    hittable* found = this->m_objects.get(name);
    //    if (found)
    //    {
    //        found = new rt::translate(found, vector);
    //    }
    //    else
    //    {
    //        // search in groups
    //        for (auto& group : this->m_groups)
    //        {
    //            hittable* found2 = group.second->get(name);
    //            if (found2)
    //            {
    //                found2 = new rt::translate(found2, vector);
    //                break;
    //            }
    //        }
    //    }
    //}
    //else
    //{
    //    hittable* back = this->m_objects.back();
    //    std::string n = back->getName();
    //    if (n == name)
    //    {
    //        hittable* tmp = this->m_objects.back();
    //        tmp = new rt::translate(back, vector);
    //    }
    //}

    return *this;
}

scene_builder& scene_builder::rotate(const vector3& vector, const char* name)
{
    //if (name != nullptr && name[0] != '\0')
    //{
    //    hittable* found = this->m_objects.get(name);
    //    if (found)
    //    {
    //        found = new rt::rotate(found, vector);
    //    }
    //    else
    //    {
    //        // search in groups
    //        for (auto& group : this->m_groups)
    //        {
    //            hittable* found2 = group.second->get(name);
    //            if (found2)
    //            {
    //                found2 = new rt::rotate(found2, vector);
    //                break;
    //            }
    //        }
    //    }
    //}
    //else
    //{
    //    hittable* back = this->m_objects.back();
    //    char* n = back->getName();
    //    if (n == name)
    //    {
    //        hittable* tmp = this->m_objects.back();
    //        tmp = new rt::rotate(back, vector);
    //    }
    //}

    return *this;
}

scene_builder& scene_builder::scale(const vector3& vector, const char* name)
{
    //if (name != nullptr && name[0] != '\0')
    //{
    //    hittable* found = this->m_objects.get(name);
    //    if (found)
    //    {
    //        found = new  rt::scale(found, vector);
    //    }
    //    else
    //    {
    //        // search in groups
    //        for (auto& group : this->m_groups)
    //        {
    //            hittable* found2 = group.second->get(name);
    //            if (found2)
    //            {
    //                found2 = new rt::scale(found2, vector);
    //                break;
    //            }
    //        }
    //    }
    //}
    //else
    //{
    //    hittable* back = this->m_objects.back();
    //    std::string n = back->getName();
    //    if (n == name)
    //    {
    //        hittable* tmp = this->m_objects.back();
    //        tmp= new rt::scale(back, vector);
    //    }
    //}

    return *this;
}

//material* scene_builder::fetchMaterial(const char* name)
//{
//    if (name != nullptr && name[0] != '\0')
//    {
//        auto it = this->m_materials.find(name);
//
//        if (it != this->m_materials.end())
//        {
//            // if key is found
//            return it->second;
//        }
//        else
//        {
//            // if key is not found
//            std::cerr << "[WARN] Material " << name << " not found !" << std::endl;
//            return nullptr;
//        }
//    }
//
//    return nullptr;
//}
//
//texture* scene_builder::fetchTexture(const char* name)
//{
//    if (name != nullptr && name[0] != '\0')
//    {
//        auto it = this->m_textures.find(name);
//
//        if (it != this->m_textures.end())
//        {
//            // if key is found
//            return it->second;
//        }
//        else
//        {
//            // if key is not found
//            std::cerr << "[WARN] Texture " << name << " not found !" << std::endl;
//            return nullptr;
//        }
//    }
//
//    return nullptr;
//}
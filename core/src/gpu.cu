#include <iostream>

// cuda
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <thrust/random.h>


#include "misc/vector3.cuh"
#include "misc/bvh_node.cuh"


#include "primitives/hittable_list.cuh"

#include "textures/texture.cuh"
#include "textures/solid_color_texture.cuh"
#include "textures/gradient_texture.cuh"
#include "textures/checker_texture.cuh"
#include "textures/image_texture.cuh"
#include "textures/bump_texture.cuh"
#include "textures/normal_texture.cuh"
#include "textures/alpha_texture.cuh"
#include "textures/emissive_texture.cuh"
#include "textures/perlin_noise_texture.cuh"



#include "materials/diffuse_light.cuh"
#include "materials/diffuse_spot_light.cuh"
#include "materials/lambertian.cuh"
#include "materials/metal.cuh"
#include "materials/dielectric.cuh"
#include "materials/isotropic.cuh"
#include "materials/anisotropic.cuh"
#include "materials/oren_nayar.cuh"
#include "materials/phong.cuh"



#include "primitives/aarect.cuh"
#include "primitives/box.cuh"
#include "primitives/sphere.cuh"
#include "primitives/quad.cuh"
#include "primitives/volume.cuh"
#include "primitives/torus.cuh"
#include "primitives/cylinder.cuh"
#include "primitives/cone.cuh"
#include "primitives/disk.cuh"
#include "primitives/triangle.cuh"

#include "primitives/translate.cuh"
#include "primitives/rotate.cuh"
#include "primitives/scale.cuh"
#include "primitives/flip_normals.cuh"

#include "utilities/mesh_loader.cuh"


#include "lights/light.cuh"
#include "lights/omni_light.cuh"
#include "lights/directional_light.cuh"
#include "lights/spot_light.cuh"

#include "cameras/camera.cuh"
#include "cameras/perspective_camera.cuh"
#include "cameras/orthographic_camera.cuh"

#include "samplers/sampler.cuh"
#include "samplers/random_sampler.cuh"

#include "utilities/bitmap_image.cuh"


#include "scenes/scene_config.h"

#include "scene_factory.cuh"



#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>


// Structure to hold the mesh data (vertices, normals, indices)
// Structure to hold mesh data for a single OBJ file
struct MeshData {
    float* d_vertices;  // Device pointer for vertices
    float* d_normals;   // Device pointer for normals
    int* d_indices;     // Device pointer for indices
    size_t num_vertices;
    size_t num_indices;
};

// Structure to hold data for multiple meshes (multiple OBJ files)
struct SceneData {
    std::vector<MeshData> meshes;  // Vector of mesh data for multiple OBJ files
};


// https://github.com/Belval/raytracing

bool isGpuAvailable(cudaDeviceProp& prop)
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaGetDeviceProperties(&prop, deviceIndex);
        if (prop.major >= 2 && prop.minor >= 0)
        {
            printf("[INFO] Use GPU device %d %s\n", deviceIndex, prop.name);
            printf("[INFO] Number of multiprocessors on device: %d\n", prop.multiProcessorCount);
            printf("[INFO] Total global memory (Gbytes) %.1f\n", (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
            printf("[INFO] Max grid size: %i x %i x %i\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            printf("[INFO] Max block size: %i x %i x %i\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            printf("[INFO] Max number of threads per block: %i\n", prop.maxThreadsPerBlock);

            cudaSetDevice(deviceIndex);

            return true;
        }
    }

    std::cout << "[ERROR] No Nvidia Cuda GPU device found" << std::endl;
    return false;
}


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' " << cudaGetErrorString(result) << "\n";
        // reset device before terminating
        cudaDeviceReset();
        exit(99);
    }
}

// move in cuda helpers
__device__ int strcmp_device(const char* str1, const char* str2)
{
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char*)str1 - *(unsigned char*)str2;
}



__device__ int getTextureImageIndex(sceneConfig* sceneCfg, bitmap_image** images, const char* textureName)
{
    int index = 0;
    
    if (textureName == nullptr || textureName[0] == '\0')
    {
        printf("[GPU] Can't get image index for nameless texture\n");
        return index;
    }

    for (int i = 0; i < sceneCfg->texturesCfg.imageTextureCount; i++)
    {
        imageTextureConfig imageTexture = sceneCfg->texturesCfg.imageTextures[i];

        if (strcmp_device(imageTexture.name, textureName) == 0)
        {
            return index;
        }

        index++;
    }

    printf("[GPU] Can't get image index for texture %s\n", textureName);

    return index;
}



/// <summary>
/// Only spheres and box boundaries for the moment !!!
/// </summary>
/// <param name="sceneCfg"></param>
/// <param name="primitiveName"></param>
/// <returns></returns>
__device__ hittable* fetchPrimitive(sceneConfig* sceneCfg, const char* primitiveName)
{
    if (primitiveName == nullptr || primitiveName[0] == '\0')
    {
        printf("[GPU] Can't fetch nameless primitive\n");
        return nullptr;
    }
    
    auto defaultMaterial = new lambertian(new solid_color_texture(1.0f, 1.0f, 1.0f));
    
    for (int i = 0; i < sceneCfg->primitivesCfg.spherePrimitiveCount; i++)
    {
        spherePrimitiveConfig spherePrimitive = sceneCfg->primitivesCfg.spherePrimitives[i];
        if (strcmp_device(spherePrimitive.name, primitiveName) == 0)
        {
            return scene_factory::createSphere(spherePrimitive.name, spherePrimitive.position, spherePrimitive.radius, defaultMaterial, spherePrimitive.mapping, spherePrimitive.transform);
        }
    }

    for (int i = 0; i < sceneCfg->primitivesCfg.boxPrimitiveCount; i++)
    {
        boxPrimitiveConfig boxPrimitive = sceneCfg->primitivesCfg.boxPrimitives[i];
        if (strcmp_device(boxPrimitive.name, primitiveName) == 0)
        {
            return scene_factory::createBox(boxPrimitive.name, boxPrimitive.position, boxPrimitive.size, defaultMaterial, boxPrimitive.mapping, boxPrimitive.transform);
        }
    }

    return nullptr;
}

__device__ texture* fetchTexture(sceneConfig* sceneCfg, bitmap_image** images, int count_images, const char* textureName)
{
    if (textureName == nullptr || textureName[0] == '\0')
    {
        printf("[GPU] Can't fetch nameless texture\n");
        return nullptr;
    }
    
    for (int i = 0; i < sceneCfg->texturesCfg.solidColorTextureCount; i++)
    {
        solidColorTextureConfig solidColorTexture = sceneCfg->texturesCfg.solidColorTextures[i];
        
        if (strcmp_device(solidColorTexture.name, textureName) == 0)
        {
            return scene_factory::createColorTexture(solidColorTexture.name, solidColorTexture.rgb);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.gradientColorTextureCount; i++)
    {
        gradientColorTextureConfig gradientColorTexture = sceneCfg->texturesCfg.gradientColorTextures[i];
        
        if (strcmp_device(gradientColorTexture.name, textureName) == 0)
        {
            return scene_factory::createGradientTexture(gradientColorTexture.name, gradientColorTexture.color1, gradientColorTexture.color2, gradientColorTexture.vertical, gradientColorTexture.hsv);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.imageTextureCount; i++)
    {
        imageTextureConfig imageTexture = sceneCfg->texturesCfg.imageTextures[i];
        
        if (strcmp_device(imageTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createImageTexture(imageTexture.name, imageTexture.filepath, img, true);
            }
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.checkerTextureCount; i++)
    {
        checkerTextureConfig checkerTexture = sceneCfg->texturesCfg.checkerTextures[i];

        if (strcmp_device(checkerTexture.name, textureName) == 0)
        {
            texture* oddTexture = fetchTexture(sceneCfg, images, count_images, checkerTexture.oddTextureName);
            texture* evenTexture = fetchTexture(sceneCfg, images, count_images, checkerTexture.evenTextureName);
            
            if (oddTexture != nullptr && evenTexture != nullptr)
                return scene_factory::createCheckerTexture(checkerTexture.name, oddTexture, checkerTexture.oddTextureName, evenTexture, checkerTexture.evenTextureName, checkerTexture.scale);
            else
                return scene_factory::createCheckerTexture(checkerTexture.name, checkerTexture.oddColor, checkerTexture.evenColor, oddTexture, evenTexture, checkerTexture.scale);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.noiseTextureCount; i++)
    {
        noiseTextureConfig noiseTexture = sceneCfg->texturesCfg.noiseTextures[i];

        if (strcmp_device(noiseTexture.name, textureName) == 0)
        {
            return scene_factory::createNoiseTexture(noiseTexture.name, noiseTexture.scale);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.bumpTextureCount; i++)
    {
        bumpTextureConfig bumpTexture = sceneCfg->texturesCfg.bumpTextures[i];


        if (strcmp_device(bumpTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createBumpTexture(bumpTexture.name, bumpTexture.filepath, img, bumpTexture.strength);
            }
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.normalTextureCount; i++)
    {
        normalTextureConfig normalTexture = sceneCfg->texturesCfg.normalTextures[i];

        if (strcmp_device(normalTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createNormalTexture(normalTexture.name, normalTexture.filepath, img, normalTexture.strength);
            }
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.displacementTextureCount; i++)
    {
        displacementTextureConfig displacementTexture = sceneCfg->texturesCfg.displacementTextures[i];

        if (strcmp_device(displacementTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createDisplaceTexture(displacementTexture.name, displacementTexture.filepath, img, displacementTexture.strength);
            }
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.alphaTextureCount; i++)
    {
        alphaTextureConfig alphaTexture = sceneCfg->texturesCfg.alphaTextures[i];

        if (strcmp_device(alphaTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createAlphaTexture(alphaTexture.name, alphaTexture.filepath, img, alphaTexture.doubleSided);
            }
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.emissiveTextureCount; i++)
    {
        emissiveTextureConfig emissiveTexture = sceneCfg->texturesCfg.emissiveTextures[i];

        if (strcmp_device(emissiveTexture.name, textureName) == 0)
        {
            int image_index = getTextureImageIndex(sceneCfg, images, textureName);
            if (image_index >= 0 && image_index < count_images)
            {
                bitmap_image img = *(images[image_index]);
                return scene_factory::createEmissiveTexture(emissiveTexture.name, emissiveTexture.filepath, img, emissiveTexture.strength);
            }
        }
    }

    return nullptr;
}


__device__ material* fetchMaterial(sceneConfig* sceneCfg, bitmap_image** images, int count_images, const char* materialName)
{
    if (materialName == nullptr || materialName[0] == '\0')
    {
        printf("[GPU] Can't fetch nameless material\n");
        return nullptr;
    }
    
    for (int i = 0; i < sceneCfg->materialsCfg.lambertianMaterialCount; i++)
    {
        lambertianMaterialConfig lambertianMaterial = sceneCfg->materialsCfg.lambertianMaterials[i];

        if (strcmp_device(lambertianMaterial.name, materialName) == 0)
        {
            if (lambertianMaterial.textureName != nullptr && lambertianMaterial.textureName[0] != '\0')
            {
                texture* tex = fetchTexture(sceneCfg, images, count_images, lambertianMaterial.textureName);
                if (tex)
                {
                    return scene_factory::createLambertianMaterial(lambertianMaterial.name, lambertianMaterial.textureName, tex);
                }
            }
            else
            {
                return scene_factory::createLambertianMaterial(lambertianMaterial.name, lambertianMaterial.rgb);
            }
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.metalMaterialCount; i++)
    {
        metalMaterialConfig metalMaterial = sceneCfg->materialsCfg.metalMaterials[i];
        
        if (strcmp_device(metalMaterial.name, materialName) == 0)
        {
            return scene_factory::createMetalMaterial(metalMaterial.name, metalMaterial.rgb, metalMaterial.fuzziness);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.dielectricMaterialCount; i++)
    {
        dielectricMaterialConfig glassMaterial = sceneCfg->materialsCfg.dielectricMaterials[i];
        
        if (strcmp_device(glassMaterial.name, materialName) == 0)
        {
            return scene_factory::createDielectricMaterial(glassMaterial.name, glassMaterial.refraction);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.isotropicMaterialCount; i++)
    {
        isotropicMaterialConfig isotropicMaterial = sceneCfg->materialsCfg.isotropicMaterials[i];

        if (strcmp_device(isotropicMaterial.name, materialName) == 0)
        {
            if (isotropicMaterial.textureName != nullptr && isotropicMaterial.textureName[0] != '\0')
            {
                texture* tex = fetchTexture(sceneCfg, images, count_images, isotropicMaterial.textureName);

                if (tex)
                    return scene_factory::createIsotropicMaterial(isotropicMaterial.name, isotropicMaterial.textureName, tex);
            }
            else
                return scene_factory::createIsotropicMaterial(isotropicMaterial.name, isotropicMaterial.rgb);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.anisotropicMaterialCount; i++)
    {
        anisotropicMaterialConfig anisotropicMaterial = sceneCfg->materialsCfg.anisotropicMaterials[i];

        if (strcmp_device(anisotropicMaterial.name, materialName) == 0)
        {
            texture* diffuse_tex = fetchTexture(sceneCfg, images, count_images, anisotropicMaterial.diffuseTextureName);
            texture* specular_tex = fetchTexture(sceneCfg, images, count_images, anisotropicMaterial.specularTextureName);
            texture* exponent_tex = fetchTexture(sceneCfg, images, count_images, anisotropicMaterial.exponentTextureName);
            
            if (diffuse_tex)
                return scene_factory::createAnisotropicMaterial(anisotropicMaterial.name, anisotropicMaterial.nuf, anisotropicMaterial.nvf, anisotropicMaterial.diffuseTextureName, diffuse_tex, anisotropicMaterial.specularTextureName, specular_tex, anisotropicMaterial.exponentTextureName, exponent_tex);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.orenNayarMaterialCount; i++)
    {
        orenNayarMaterialConfig orenNayarMaterial = sceneCfg->materialsCfg.orenNayarMaterials[i];

        if (strcmp_device(orenNayarMaterial.name, materialName) == 0)
        {
            texture* tex = fetchTexture(sceneCfg, images, count_images, orenNayarMaterial.textureName);

            if (tex)
                return scene_factory::createOrenNayarMaterial(orenNayarMaterial.name, orenNayarMaterial.rgb, orenNayarMaterial.roughness, orenNayarMaterial.albedo_temp);
            else
                return scene_factory::createOrenNayarMaterial(orenNayarMaterial.name, orenNayarMaterial.textureName, tex, orenNayarMaterial.roughness, orenNayarMaterial.albedo_temp);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.phongMaterialCount; i++)
    {
        phongMaterialConfig phongMaterial = sceneCfg->materialsCfg.phongMaterials[i];

        if (strcmp_device(phongMaterial.name, materialName) == 0)
        {
            texture* diffuse_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.diffuseTextureName);
            texture* specular_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.specularTextureName);
            texture* bump_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.bumpTextureName);
            texture* normal_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.normalTextureName);
            texture* displace_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.displacementTextureName);
            texture* alpha_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.alphaTextureName);
            texture* emissive_tex = fetchTexture(sceneCfg, images, count_images, phongMaterial.emissiveTextureName);
            
            if (diffuse_tex)
                return scene_factory::createPhongMaterial(phongMaterial.name, phongMaterial.diffuseTextureName, diffuse_tex, phongMaterial.specularTextureName, specular_tex, phongMaterial.bumpTextureName, bump_tex, phongMaterial.normalTextureName, normal_tex, phongMaterial.displacementTextureName, displace_tex, phongMaterial.alphaTextureName, alpha_tex, phongMaterial.emissiveTextureName, emissive_tex, phongMaterial.ambientColor, phongMaterial.shininess);
        }
    }

    return nullptr;
}




__global__ void load_scene(sceneConfig* sceneCfg, hittable_list **elist, hittable_list **elights, camera **cam, sampler **aa_sampler, int width, int height, float ratio, int spp, int sqrt_spp, bitmap_image** images, int count_images, int seed)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // thrust random engine and distribution
        thrust::minstd_rand rng(seed);
        thrust::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);


        *elights = new hittable_list();

        *elist = new hittable_list();




        // LIGHTS
        printf("[GPU] %i omni lights found\n", sceneCfg->lightsCfg.omniLightCount);
        printf("[GPU] %i dir lights found\n", sceneCfg->lightsCfg.dirLightCount);
        printf("[GPU] %i spot lights found\n", sceneCfg->lightsCfg.spotLightCount);

        for (int i = 0; i < sceneCfg->lightsCfg.omniLightCount; i++)
        {
            omniLightConfig omnilight = sceneCfg->lightsCfg.omniLights[i];

            (*elist)->add(scene_factory::createOmniLight(omnilight.name, omnilight.position, omnilight.radius, omnilight.intensity, omnilight.rgb, omnilight.invisible, true));
        }

        for (int i = 0; i < sceneCfg->lightsCfg.dirLightCount; i++)
        {
            directionalLightConfig dirlight = sceneCfg->lightsCfg.dirLights[i];

            (*elist)->add(scene_factory::createDirLight(dirlight.name, dirlight.position, dirlight.u, dirlight.v, dirlight.intensity, dirlight.rgb, dirlight.invisible, true));
        }

        for (int i = 0; i < sceneCfg->lightsCfg.spotLightCount; i++)
        {
            spotLightConfig spotlight = sceneCfg->lightsCfg.spotLights[i];

            (*elist)->add(scene_factory::createSpotLight(spotlight.name, spotlight.position, spotlight.direction, spotlight.cutoff, spotlight.falloff, spotlight.intensity, spotlight.radius, spotlight.rgb, spotlight.invisible, true));
        }


        // PRIMITIVES
        for (int i = 0; i < sceneCfg->primitivesCfg.spherePrimitiveCount; i++)
        {
            spherePrimitiveConfig spherePrimitive = sceneCfg->primitivesCfg.spherePrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, spherePrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createSphere(spherePrimitive.name, spherePrimitive.position, spherePrimitive.radius, mat, spherePrimitive.mapping, spherePrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.planePrimitiveCount; i++)
        {
            planePrimitiveConfig planePrimitive = sceneCfg->primitivesCfg.planePrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, planePrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createPlane(planePrimitive.name, planePrimitive.point1, planePrimitive.point2, mat, planePrimitive.mapping, planePrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.quadPrimitiveCount; i++)
        {
            quadPrimitiveConfig quadPrimitive = sceneCfg->primitivesCfg.quadPrimitives[i];
            
            material* mat = fetchMaterial(sceneCfg, images, count_images, quadPrimitive.materialName);
            
            if (mat)
                (*elist)->add(scene_factory::createQuad(quadPrimitive.name, quadPrimitive.position, quadPrimitive.u, quadPrimitive.v, mat, quadPrimitive.mapping, quadPrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.boxPrimitiveCount; i++)
        {
            boxPrimitiveConfig boxPrimitive = sceneCfg->primitivesCfg.boxPrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, boxPrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createBox(boxPrimitive.name, boxPrimitive.position, boxPrimitive.size, mat, boxPrimitive.mapping, boxPrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.conePrimitiveCount; i++)
        {
            conePrimitiveConfig conePrimitive = sceneCfg->primitivesCfg.conePrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, conePrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createCone(conePrimitive.name, conePrimitive.position, conePrimitive.height, conePrimitive.radius, mat, conePrimitive.mapping, conePrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.cylinderPrimitiveCount; i++)
        {
            cylinderPrimitiveConfig cylinderPrimitive = sceneCfg->primitivesCfg.cylinderPrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, cylinderPrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createCylinder(cylinderPrimitive.name, cylinderPrimitive.position, cylinderPrimitive.height, cylinderPrimitive.radius, mat, cylinderPrimitive.mapping, cylinderPrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.diskPrimitiveCount; i++)
        {
            diskPrimitiveConfig diskPrimitive = sceneCfg->primitivesCfg.diskPrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, diskPrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createDisk(diskPrimitive.name, diskPrimitive.position, diskPrimitive.height, diskPrimitive.radius, mat, diskPrimitive.mapping, diskPrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.torusPrimitiveCount; i++)
        {
            torusPrimitiveConfig torusPrimitive = sceneCfg->primitivesCfg.torusPrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, torusPrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createTorus(torusPrimitive.name, torusPrimitive.position, torusPrimitive.major_radius, torusPrimitive.minor_radius, mat, torusPrimitive.mapping, torusPrimitive.transform));
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.volumePrimitiveCount; i++)
        {
            volumePrimitiveConfig volumePrimitive = sceneCfg->primitivesCfg.volumePrimitives[i];

            hittable* boundary = fetchPrimitive(sceneCfg, volumePrimitive.boundaryName);
            texture* tex = fetchTexture(sceneCfg, images, count_images, volumePrimitive.textureName);
            
            if (boundary && tex)
                (*elist)->add(scene_factory::createVolume(volumePrimitive.name, boundary, volumePrimitive.density, volumePrimitive.rgb, tex, volumePrimitive.transform));
        }


        // MESHES
        for (int i = 0; i < sceneCfg->meshesCfg.objMeshCount; i++)
        {
            objMeshConfig objMesh = sceneCfg->meshesCfg.objMeshes[i];

            material* mat = fetchMaterial(sceneCfg, images, count_images, objMesh.materialName);

            if (mat)
                (*elist)->add(scene_factory::createObjMesh(objMesh.name, objMesh.position, objMesh.filepath, mat, objMesh.use_mtl, objMesh.use_smoothing, objMesh.transform));
        }



        // temp extract_emissive_objects
        for (int i = 0; i < (*elist)->object_count; i++)
        {
            if ((*elist)->objects[i]->getTypeID() == HittableTypeID::lightType
                || (*elist)->objects[i]->getTypeID() == HittableTypeID::lightDirectionalType
                || (*elist)->objects[i]->getTypeID() == HittableTypeID::lightSpotType
                || (*elist)->objects[i]->getTypeID() == HittableTypeID::lightOmniType)
            {
                light* derived = static_cast<light*>((*elist)->objects[i]);
                if (derived)
                {
                    (*elights)->add((*elist)->objects[i]);
                }
            }
        }

        if (!sceneCfg->cameraCfg.isOrthographic)
        {
            *cam = new perspective_camera();
            (*cam)->vfov = sceneCfg->cameraCfg.fov;
        }
        else
        {
            *cam = new orthographic_camera();
            (*cam)->ortho_height = sceneCfg->cameraCfg.orthoHeight;
        }

        (*cam)->initialize(
            sceneCfg->cameraCfg.lookFrom,
            sceneCfg->cameraCfg.lookAt,
            sceneCfg->cameraCfg.upAxis,
            sceneCfg->imageCfg.width,
            sceneCfg->cameraCfg.aspectRatio,
            sceneCfg->cameraCfg.fov, // perspective_camera only
            sceneCfg->cameraCfg.aperture,
            sceneCfg->cameraCfg.focus,
            sceneCfg->cameraCfg.orthoHeight, // orthographic_camera only
            0.0f,
            1.0f,
            sqrt_spp);



        (*cam)->samples_per_pixel = sceneCfg->imageCfg.spp; // denoiser quality
        (*cam)->max_depth = sceneCfg->imageCfg.depth; // max nbr of bounces a ray can do
        (*cam)->background_color = color(0.70f, 0.80f, 1.00f);


    
        // Background
        //if (sceneCfg->imageCfg.background.filepath != nullptr)
        //{
        //    //auto background = new image_texture(imageCfg.background.filepath);
        //    //cam->background_texture = background;
        //    //cam->background_iskybox = imageCfg.background.is_skybox;

        //    //if (imageCfg.background.is_skybox)
        //    //    cam->background_pdf = new image_pdf(background);
        //}
        //else
        //{
            (*cam)->background_color = sceneCfg->imageCfg.background.rgb;
        //}





        // calculate bounding boxes to speed up ray computing
        *elist = new hittable_list(new bvh_node((*elist)->objects, 0, (*elist)->object_count, rng));
    }
}

__global__ void image_init(unsigned char* tex_data, int width, int height, int channels, bitmap_image** tex)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *tex = new bitmap_image(tex_data, width, height, channels);
    }
}

__global__ void render(color* fb, int width, int height, int spp, int sqrt_spp, int max_depth, hittable_list** world, hittable_list** lights, camera** cam, sampler** aa_sampler, int seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= width) || (j >= height)) return;

    int pixel_index = j * width + i;

    // Initialize the random engine and distribution
    thrust::minstd_rand rng(seed + pixel_index);
    thrust::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);


    color pixel_color(0, 0, 0);

    for (int s_j = 0; s_j < sqrt_spp; ++s_j)
    {
        for (int s_i = 0; s_i < sqrt_spp; ++s_i)
        {
            // Generate a random value between 0 and 1
            float uniform_random = uniform_dist(rng);

            // Stratified sampling within the pixel, with Sobol randomness
            float u = (i + (s_i + uniform_random) / sqrt_spp) / float(width);
            float v = (j + (s_j + uniform_random) / sqrt_spp) / float(height);

            ray r = (*cam)->get_ray(u, v, s_i, s_j, nullptr, rng);
            pixel_color += (*cam)->ray_color(r, i, j, max_depth, max_depth, **world, **lights, rng);
        }
    }

    const color& fix = prepare_pixel_color(i, j, pixel_color, spp, true);
    const interval intensity(0.000f, 0.999f);

    int color_r = static_cast<int>(255.99f * intensity.clamp(fix.r()));
    int color_g = static_cast<int>(255.99f * intensity.clamp(fix.g()));
    int color_b = static_cast<int>(255.99f * intensity.clamp(fix.b()));

    fb[pixel_index] = color(color_r, color_g, color_b);

    printf("p %u %u %u %u %u\n", i, height - j - 1, color_r, color_g, color_b);
}


void setupCuda(const cudaDeviceProp& prop)
{
    // If you get a null pointer (either from device malloc or device new) you have run out of heap space.
    // https://forums.developer.nvidia.com/t/allocating-memory-from-device-and-cudalimitmallocheapsize/70441
    
    size_t stackSize;

    // Get the current stack size limit
    cudaError_t result1 = cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    if (result1 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to get stack size: " << cudaGetErrorString(result1) << std::endl;
        return;
    }

    std::cout << "[INFO] Current stack size limit: " << stackSize << " bytes" << std::endl;


    const size_t newStackSize = 4096; // Set the stack size to 1MB per thread

    cudaError_t result2 = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (result2 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set stack size: " << cudaGetErrorString(result2) << std::endl;
        return;
    }

    std::cout << "[INFO] New stack size limit: " << newStackSize << " bytes" << std::endl;



    const size_t newMallocHeapSize = size_t(1024) * size_t(1024) * size_t(1024);

    cudaError_t result3 = cudaDeviceSetLimit(cudaLimitMallocHeapSize, newMallocHeapSize);
    if (result3 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set malloc heap size: " << cudaGetErrorString(result3) << std::endl;
        return;
    }

    std::cout << "[INFO] New malloc heap limit: " << newMallocHeapSize << " bytes" << std::endl;



    const size_t newPrintfFifoSize = 250000000;

    cudaError_t result4 = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, newPrintfFifoSize);
    if (result4 != cudaSuccess) {
        std::cerr << "[WARNING] Failed to set printf fifo size: " << cudaGetErrorString(result4) << std::endl;
        return;
    }

    std::cout << "[INFO] New printf fifo size: " << newPrintfFifoSize << " bytes" << std::endl;
}

/// <summary>
/// Helper function to copy a string from the host to the device
/// </summary>
/// <param name="hostString">Pointer to the host string</param>
/// <param name="deviceString">Pointer to the device string (output)</param>
void copyStringToDevice(const char* hostString, char** deviceString)
{
    // Allocate memory on the device for the string (with null terminator)
    size_t stringLen = strlen(hostString) + 1;  // +1 for null terminator
    cudaMalloc((void**)deviceString, stringLen);

    // Copy the string from host to device
    cudaMemcpy(*deviceString, hostString, stringLen, cudaMemcpyHostToDevice);
}


/// <summary>
/// Helper function to copy texture configuration
/// </summary>
template<typename TextureConfig>
void copyCommonTextureConfig(const TextureConfig* h_textures, int count, TextureConfig** d_textures, texturesConfig* d_texturesCfg, TextureConfig** d_texturesPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_textures, count * sizeof(TextureConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_textures, h_textures, count * sizeof(TextureConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_textures[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_textures)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_texturesPtrOnDevice, d_textures, sizeof(TextureConfig*), cudaMemcpyHostToDevice);
}




/// <summary>
/// // Helper function to copy common material configuration
/// </summary>
template<typename MaterialConfig>
void copyCommonMaterialConfig(const MaterialConfig* h_materials, int count, MaterialConfig** d_materials, materialsConfig* d_materialsCfg, MaterialConfig** d_materialsPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_materials, count * sizeof(MaterialConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_materials, h_materials, count * sizeof(MaterialConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_materials[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_materialsPtrOnDevice, d_materials, sizeof(MaterialConfig*), cudaMemcpyHostToDevice);
}

/// <summary>
/// // Helper function to copy common material configuration
/// </summary>
template<typename MaterialConfig>
void copyTextureMaterialConfig(const MaterialConfig* h_materials, int count, MaterialConfig** d_materials, materialsConfig* d_materialsCfg, MaterialConfig** d_materialsPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_materials, count * sizeof(MaterialConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_materials, h_materials, count * sizeof(MaterialConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_materials[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy textureName
        const char* hostTextureName = h_materials[i].textureName;  // Get the string from the host
        char* d_textureName;
        copyStringToDevice(hostTextureName, &d_textureName);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].textureName), &d_textureName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_materialsPtrOnDevice, d_materials, sizeof(MaterialConfig*), cudaMemcpyHostToDevice);
}


/// <summary>
/// // Helper function to copy anisotropic material configuration
/// </summary>
template<typename MaterialConfig>
void copyAnisotropicMaterialConfig(const MaterialConfig* h_materials, int count, MaterialConfig** d_materials, materialsConfig* d_materialsCfg, MaterialConfig** d_materialsPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_materials, count * sizeof(MaterialConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_materials, h_materials, count * sizeof(MaterialConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_materials[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy diffuseTextureName
        const char* hostDiffuseTextureName = h_materials[i].diffuseTextureName;  // Get the string from the host
        char* d_diffuseTextureName;
        copyStringToDevice(hostDiffuseTextureName, &d_diffuseTextureName);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].diffuseTextureName), &d_diffuseTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy diffuseTextureName
        const char* hostSpecularTextureName = h_materials[i].specularTextureName;  // Get the string from the host
        char* d_specularTextureName;
        copyStringToDevice(hostSpecularTextureName, &d_specularTextureName);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].specularTextureName), &d_specularTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy exponentTextureName
        const char* hostExponentTextureName = h_materials[i].exponentTextureName;  // Get the string from the host
        char* d_exponentTextureName;
        copyStringToDevice(hostExponentTextureName, &d_exponentTextureName);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].exponentTextureName), &d_exponentTextureName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_materialsPtrOnDevice, d_materials, sizeof(MaterialConfig*), cudaMemcpyHostToDevice);
}

/// <summary>
/// // Helper function to copy phong material configuration
/// </summary>
template<typename MaterialConfig>
void copyPhongMaterialConfig(const MaterialConfig* h_materials, int count, MaterialConfig** d_materials, materialsConfig* d_materialsCfg, MaterialConfig** d_materialsPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_materials, count * sizeof(MaterialConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_materials, h_materials, count * sizeof(MaterialConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_materials[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_materials)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy diffuseTextureName
        const char* hostDiffuseTextureName = h_materials[i].diffuseTextureName;  // Get the string from the host
        char* d_diffuseTextureName;
        copyStringToDevice(hostDiffuseTextureName, &d_diffuseTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].diffuseTextureName), &d_diffuseTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy specularTextureName
        const char* hostSpecularTextureName = h_materials[i].specularTextureName;  // Get the string from the host
        char* d_specularTextureName;
        copyStringToDevice(hostSpecularTextureName, &d_specularTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].specularTextureName), &d_specularTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy bumpTextureName
        const char* hostBumpTextureName = h_materials[i].bumpTextureName;  // Get the string from the host
        char* d_bumpTextureName;
        copyStringToDevice(hostBumpTextureName, &d_bumpTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].bumpTextureName), &d_bumpTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy normalTextureName
        const char* hostNormalTextureName = h_materials[i].normalTextureName;  // Get the string from the host
        char* d_normalTextureName;
        copyStringToDevice(hostNormalTextureName, &d_normalTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].normalTextureName), &d_normalTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy displacementTextureName
        const char* hostDisplacementTextureName = h_materials[i].displacementTextureName;  // Get the string from the host
        char* d_displacementTextureName;
        copyStringToDevice(hostDisplacementTextureName, &d_displacementTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].displacementTextureName), &d_displacementTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy alphaTextureName
        const char* hostAlphaTextureName = h_materials[i].alphaTextureName;  // Get the string from the host
        char* d_alphaTextureName;
        copyStringToDevice(hostAlphaTextureName, &d_alphaTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].alphaTextureName), &d_alphaTextureName, sizeof(char*), cudaMemcpyHostToDevice);


        // Copy emissiveTextureName
        const char* hostEmissiveTextureName = h_materials[i].emissiveTextureName;  // Get the string from the host
        char* d_emissiveTextureName;
        copyStringToDevice(hostEmissiveTextureName, &d_emissiveTextureName);  // Use reusable function
        cudaMemcpy(&((*d_materials)[i].emissiveTextureName), &d_emissiveTextureName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_materialsPtrOnDevice, d_materials, sizeof(MaterialConfig*), cudaMemcpyHostToDevice);
}




/// <summary>
/// // Helper function to copy common texture configuration
/// </summary>
template<typename TextureConfig>
void copyImageTextureConfig(const TextureConfig* h_textures, int count, TextureConfig** d_textures, texturesConfig* d_texturesCfg, TextureConfig** d_texturesPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_textures, count * sizeof(TextureConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_textures, h_textures, count * sizeof(TextureConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_textures[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_textures)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy filepath
        const char* hostFilepath = h_textures[i].filepath;  // Get the string from the host
        char* d_filepath;
        copyStringToDevice(hostFilepath, &d_filepath);  // Use reusable function for filepath
        cudaMemcpy(&((*d_textures)[i].filepath), &d_filepath, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_texturesPtrOnDevice, d_textures, sizeof(TextureConfig*), cudaMemcpyHostToDevice);
}

/// <summary>
/// // Helper function to copy common primitive configuration
/// </summary>
template<typename PrimitiveConfig>
void copyCommonPrimitiveConfig(const PrimitiveConfig* h_primitives, int count, PrimitiveConfig** d_primitives, primitivesConfig* d_primitivesCfg, PrimitiveConfig** d_primitivesPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_primitives, count * sizeof(PrimitiveConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_primitives, h_primitives, count * sizeof(PrimitiveConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each primitive
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_primitives[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_primitives)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy materialName
        const char* hostMaterialName = h_primitives[i].materialName;  // Get the string from the host
        char* d_materialName;
        copyStringToDevice(hostMaterialName, &d_materialName);  // Use reusable function for material name
        cudaMemcpy(&((*d_primitives)[i].materialName), &d_materialName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy groupName
        const char* hostGroupName = h_primitives[i].groupName;  // Get the string from the host
        char* d_groupName;
        copyStringToDevice(hostGroupName, &d_groupName);  // Use reusable function for group name
        cudaMemcpy(&((*d_primitives)[i].groupName), &d_groupName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_primitivesPtrOnDevice, d_primitives, sizeof(PrimitiveConfig*), cudaMemcpyHostToDevice);
}



/// <summary>
/// // Helper function to copy volume primitive configuration
/// </summary>
template<typename PrimitiveConfig>
void copyVolumePrimitiveConfig(const PrimitiveConfig* h_primitives, int count, PrimitiveConfig** d_primitives, primitivesConfig* d_primitivesCfg, PrimitiveConfig** d_primitivesPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_primitives, count * sizeof(PrimitiveConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_primitives, h_primitives, count * sizeof(PrimitiveConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each primitive
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_primitives[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_primitives)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy boundaryObjectName
        const char* hostBoundaryName = h_primitives[i].boundaryName;  // Get the string from the host
        char* d_boundaryName;
        copyStringToDevice(hostBoundaryName, &d_boundaryName);  // Use reusable function for boundary name
        cudaMemcpy(&((*d_primitives)[i].boundaryName), &d_boundaryName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy textureName
        const char* hostTextureName = h_primitives[i].textureName;  // Get the string from the host
        char* d_textureName;
        copyStringToDevice(hostTextureName, &d_textureName);  // Use reusable function for texture name
        cudaMemcpy(&((*d_primitives)[i].textureName), &d_textureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy groupName
        const char* hostGroupName = h_primitives[i].groupName;  // Get the string from the host
        char* d_groupName;
        copyStringToDevice(hostGroupName, &d_groupName);  // Use reusable function for group name
        cudaMemcpy(&((*d_primitives)[i].groupName), &d_groupName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_primitivesPtrOnDevice, d_primitives, sizeof(PrimitiveConfig*), cudaMemcpyHostToDevice);
}


/// <summary>
/// // Helper function to copy texture configuration
/// </summary>
template<typename TextureConfig>
void copyCheckerTextureConfig(const TextureConfig* h_textures, int count, TextureConfig** d_textures, texturesConfig* d_texturesCfg, TextureConfig** d_texturesPtrOnDevice)
{
    // 1. Allocate memory for the array on the device
    cudaMalloc((void**)d_textures, count * sizeof(TextureConfig));

    // 2. Copy the array contents from host to device
    cudaMemcpy(*d_textures, h_textures, count * sizeof(TextureConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each texture
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_textures[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_textures)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy oddTextureName
        const char* hostOddTextureName = h_textures[i].oddTextureName;  // Get the string from the host
        char* d_oddTextureName;
        copyStringToDevice(hostOddTextureName, &d_oddTextureName);  // Use reusable function for filepath
        cudaMemcpy(&((*d_textures)[i].oddTextureName), &d_oddTextureName, sizeof(char*), cudaMemcpyHostToDevice);

        // Copy evenTextureName
        const char* hostEvenTextureName = h_textures[i].evenTextureName;  // Get the string from the host
        char* d_evenTextureName;
        copyStringToDevice(hostEvenTextureName, &d_evenTextureName);  // Use reusable function for filepath
        cudaMemcpy(&((*d_textures)[i].evenTextureName), &d_evenTextureName, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side config to point to the array on the device
    cudaMemcpy(d_texturesPtrOnDevice, d_textures, sizeof(TextureConfig*), cudaMemcpyHostToDevice);
}


/// <summary>
/// Helper function to copy light configuration
/// </summary>
template<typename LightConfig>
void copyLightConfig(const LightConfig* h_lights, int count, LightConfig** d_lights, lightsConfig* d_lightsCfg, LightConfig** d_lightsPtrOnDevice)
{
    // 1. Allocate memory for the lights array on the device
    cudaMalloc((void**)d_lights, count * sizeof(LightConfig));

    // 2. Copy the lights array contents from host to device
    cudaMemcpy(*d_lights, h_lights, count * sizeof(LightConfig), cudaMemcpyHostToDevice);

    // 3. Allocate memory and copy the names for each light
    for (int i = 0; i < count; i++)
    {
        // Copy name
        const char* hostName = h_lights[i].name;  // Get the string from the host
        char* d_name;
        copyStringToDevice(hostName, &d_name);  // Use reusable function for name
        cudaMemcpy(&((*d_lights)[i].name), &d_name, sizeof(char*), cudaMemcpyHostToDevice);
    }

    // 4. Update the device-side lightsConfig to point to the lights array on the device
    cudaMemcpy(d_lightsPtrOnDevice, d_lights, sizeof(LightConfig*), cudaMemcpyHostToDevice);
}

texturesConfig* prepareTextures(const texturesConfig& h_texturesCfg)
{
    // Allocate and copy the textures data (for solid color, gradient color, image...)
    texturesConfig* d_texturesCfg;
    cudaMalloc((void**)&d_texturesCfg, sizeof(texturesConfig));

    // Solid color textures
    if (h_texturesCfg.solidColorTextureCount > 0)
    {
        solidColorTextureConfig* d_solidColorTextures;
        copyCommonTextureConfig(h_texturesCfg.solidColorTextures, h_texturesCfg.solidColorTextureCount, &d_solidColorTextures, d_texturesCfg, &(d_texturesCfg->solidColorTextures));
    }

    // Copy the scalar values from host to device for solid color texture count
    cudaMemcpy(&(d_texturesCfg->solidColorTextureCount), &(h_texturesCfg.solidColorTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Gradient color textures
    if (h_texturesCfg.gradientColorTextureCount > 0)
    {
        gradientColorTextureConfig* d_gradientColorTextures;
        copyCommonTextureConfig(h_texturesCfg.gradientColorTextures, h_texturesCfg.gradientColorTextureCount, &d_gradientColorTextures, d_texturesCfg, &(d_texturesCfg->gradientColorTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->gradientColorTextureCount), &(h_texturesCfg.gradientColorTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Image textures
    if (h_texturesCfg.imageTextureCount > 0)
    {
        imageTextureConfig* d_imageTextures;
        copyImageTextureConfig(h_texturesCfg.imageTextures, h_texturesCfg.imageTextureCount, &d_imageTextures, d_texturesCfg, &(d_texturesCfg->imageTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->imageTextureCount), &(h_texturesCfg.imageTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Checker textures
    if (h_texturesCfg.checkerTextureCount > 0)
    {
        checkerTextureConfig* d_checkerTextures;
        copyCheckerTextureConfig(h_texturesCfg.checkerTextures, h_texturesCfg.checkerTextureCount, &d_checkerTextures, d_texturesCfg, &(d_texturesCfg->checkerTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->checkerTextureCount), &(h_texturesCfg.checkerTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Noise textures
    if (h_texturesCfg.noiseTextureCount > 0)
    {
        noiseTextureConfig* d_noiseTextures;
        copyCommonTextureConfig(h_texturesCfg.noiseTextures, h_texturesCfg.noiseTextureCount, &d_noiseTextures, d_texturesCfg, &(d_texturesCfg->noiseTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->noiseTextureCount), &(h_texturesCfg.noiseTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Bump textures
    if (h_texturesCfg.bumpTextureCount > 0)
    {
        bumpTextureConfig* d_bumpTextures;
        copyImageTextureConfig(h_texturesCfg.bumpTextures, h_texturesCfg.bumpTextureCount, &d_bumpTextures, d_texturesCfg, &(d_texturesCfg->bumpTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->bumpTextureCount), &(h_texturesCfg.bumpTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Normal textures
    if (h_texturesCfg.normalTextureCount > 0)
    {
        normalTextureConfig* d_normalTextures;
        copyImageTextureConfig(h_texturesCfg.normalTextures, h_texturesCfg.normalTextureCount, &d_normalTextures, d_texturesCfg, &(d_texturesCfg->normalTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->normalTextureCount), &(h_texturesCfg.normalTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Displacement textures
    if (h_texturesCfg.displacementTextureCount > 0)
    {
        displacementTextureConfig* d_displacementTextures;
        copyImageTextureConfig(h_texturesCfg.displacementTextures, h_texturesCfg.displacementTextureCount, &d_displacementTextures, d_texturesCfg, &(d_texturesCfg->displacementTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->displacementTextureCount), &(h_texturesCfg.displacementTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Alpha textures
    if (h_texturesCfg.alphaTextureCount > 0)
    {
        alphaTextureConfig* d_alphaTextures;
        copyImageTextureConfig(h_texturesCfg.alphaTextures, h_texturesCfg.alphaTextureCount, &d_alphaTextures, d_texturesCfg, &(d_texturesCfg->alphaTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->alphaTextureCount), &(h_texturesCfg.alphaTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Emissive textures
    if (h_texturesCfg.emissiveTextureCount > 0)
    {
        emissiveTextureConfig* d_emissiveTextures;
        copyImageTextureConfig(h_texturesCfg.emissiveTextures, h_texturesCfg.emissiveTextureCount, &d_emissiveTextures, d_texturesCfg, &(d_texturesCfg->emissiveTextures));
    }

    // Copy the scalar values from host to device for gradient color texture count
    cudaMemcpy(&(d_texturesCfg->emissiveTextureCount), &(h_texturesCfg.emissiveTextureCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    return d_texturesCfg;
}

materialsConfig* prepareMaterials(const materialsConfig& h_materialsCfg)
{
    // Allocate and copy the materials data
    materialsConfig* d_materialsCfg;
    cudaMalloc((void**)&d_materialsCfg, sizeof(materialsConfig));


    // Lambertian materials
    if (h_materialsCfg.lambertianMaterialCount > 0)
    {
        lambertianMaterialConfig* d_lambertianMaterials;
        copyTextureMaterialConfig(h_materialsCfg.lambertianMaterials, h_materialsCfg.lambertianMaterialCount, &d_lambertianMaterials, d_materialsCfg, &(d_materialsCfg->lambertianMaterials));
    }

    // Copy the scalar values from host to device for lambertian materials count
    cudaMemcpy(&(d_materialsCfg->lambertianMaterialCount), &(h_materialsCfg.lambertianMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Metal materials
    if (h_materialsCfg.metalMaterialCount > 0)
    {
        metalMaterialConfig* d_metalMaterials;
        copyCommonMaterialConfig(h_materialsCfg.metalMaterials, h_materialsCfg.metalMaterialCount, &d_metalMaterials, d_materialsCfg, &(d_materialsCfg->metalMaterials));
    }

    // Copy the scalar values from host to device for metal materials count
    cudaMemcpy(&(d_materialsCfg->metalMaterialCount), &(h_materialsCfg.metalMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Glass materials
    if (h_materialsCfg.dielectricMaterialCount > 0)
    {
        dielectricMaterialConfig* d_dielectricMaterials;
        copyCommonMaterialConfig(h_materialsCfg.dielectricMaterials, h_materialsCfg.dielectricMaterialCount, &d_dielectricMaterials, d_materialsCfg, &(d_materialsCfg->dielectricMaterials));
    }

    // Copy the scalar values from host to device for dielectric materials count
    cudaMemcpy(&(d_materialsCfg->dielectricMaterialCount), &(h_materialsCfg.dielectricMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Isotropic materials
    if (h_materialsCfg.isotropicMaterialCount > 0)
    {
        isotropicMaterialConfig* d_isotropicMaterials;
        copyTextureMaterialConfig(h_materialsCfg.isotropicMaterials, h_materialsCfg.isotropicMaterialCount, &d_isotropicMaterials, d_materialsCfg, &(d_materialsCfg->isotropicMaterials));
    }

    // Copy the scalar values from host to device for isotropic materials count
    cudaMemcpy(&(d_materialsCfg->isotropicMaterialCount), &(h_materialsCfg.isotropicMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Anisotropic materials
    if (h_materialsCfg.anisotropicMaterialCount > 0)
    {
        anisotropicMaterialConfig* d_anisotropicMaterials;
        copyAnisotropicMaterialConfig(h_materialsCfg.anisotropicMaterials, h_materialsCfg.anisotropicMaterialCount, &d_anisotropicMaterials, d_materialsCfg, &(d_materialsCfg->anisotropicMaterials));
    }

    // Copy the scalar values from host to device for anisotropic materials count
    cudaMemcpy(&(d_materialsCfg->anisotropicMaterialCount), &(h_materialsCfg.anisotropicMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Oren nayar materials
    if (h_materialsCfg.orenNayarMaterialCount > 0)
    {
        orenNayarMaterialConfig* d_orenNayarMaterials;
        copyTextureMaterialConfig(h_materialsCfg.orenNayarMaterials, h_materialsCfg.orenNayarMaterialCount, &d_orenNayarMaterials, d_materialsCfg, &(d_materialsCfg->orenNayarMaterials));
    }

    // Copy the scalar values from host to device for oren nayar materials count
    cudaMemcpy(&(d_materialsCfg->orenNayarMaterialCount), &(h_materialsCfg.orenNayarMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Phong materials
    if (h_materialsCfg.phongMaterialCount > 0)
    {
        phongMaterialConfig* d_phongMaterials;
        copyPhongMaterialConfig(h_materialsCfg.phongMaterials, h_materialsCfg.phongMaterialCount, &d_phongMaterials, d_materialsCfg, &(d_materialsCfg->phongMaterials));
    }

    // Copy the scalar values from host to device for phong materials count
    cudaMemcpy(&(d_materialsCfg->phongMaterialCount), &(h_materialsCfg.phongMaterialCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    return d_materialsCfg;
}


// Main function to prepare lights
lightsConfig* prepareLights(const lightsConfig& h_lightsCfg)
{
    // Allocate and copy the lights data (for omniLights, dirLights, spotLights)
    lightsConfig* d_lightsCfg;
    cudaMalloc((void**)&d_lightsCfg, sizeof(lightsConfig));

    // Copy omniLights array if there are omni lights
    if (h_lightsCfg.omniLightCount > 0)
    {
        omniLightConfig* d_omniLights;
        copyLightConfig(h_lightsCfg.omniLights, h_lightsCfg.omniLightCount, &d_omniLights, d_lightsCfg, &(d_lightsCfg->omniLights));
    }

    // Copy the scalar values (like omniLightCount) from host to device
    cudaMemcpy(&(d_lightsCfg->omniLightCount), &(h_lightsCfg.omniLightCount), sizeof(int8_t), cudaMemcpyHostToDevice);

    // Copy dirLights array if there are directional lights
    if (h_lightsCfg.dirLightCount > 0)
    {
        directionalLightConfig* d_dirLights;
        copyLightConfig(h_lightsCfg.dirLights, h_lightsCfg.dirLightCount, &d_dirLights, d_lightsCfg, &(d_lightsCfg->dirLights));
    }

    // Copy the scalar values (like dirLightCount) from host to device
    cudaMemcpy(&(d_lightsCfg->dirLightCount), &(h_lightsCfg.dirLightCount), sizeof(int8_t), cudaMemcpyHostToDevice);

    // Copy spotLights array if there are spot lights
    if (h_lightsCfg.spotLightCount > 0)
    {
        spotLightConfig* d_spotLights;
        copyLightConfig(h_lightsCfg.spotLights, h_lightsCfg.spotLightCount, &d_spotLights, d_lightsCfg, &(d_lightsCfg->spotLights));
    }

    // Copy the scalar values (like spotLightCount) from host to device
    cudaMemcpy(&(d_lightsCfg->spotLightCount), &(h_lightsCfg.spotLightCount), sizeof(int8_t), cudaMemcpyHostToDevice);

    return d_lightsCfg;
}

// Main function to prepare primitives
primitivesConfig* preparePrimitives(const primitivesConfig& h_primitivesCfg)
{
    primitivesConfig* d_primitivesCfg;
    cudaMalloc((void**)&d_primitivesCfg, sizeof(primitivesConfig));


    // Sphere primitives
    if (h_primitivesCfg.spherePrimitiveCount > 0)
    {
        spherePrimitiveConfig* d_spherePrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.spherePrimitives, h_primitivesCfg.spherePrimitiveCount, &d_spherePrimitives, d_primitivesCfg, &(d_primitivesCfg->spherePrimitives));
    }

    // Copy the scalar values from host to device for sphere primitives count
    cudaMemcpy(&(d_primitivesCfg->spherePrimitiveCount), &(h_primitivesCfg.spherePrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);



    // Plane primitives
    if (h_primitivesCfg.planePrimitiveCount > 0)
    {
        planePrimitiveConfig* d_planePrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.planePrimitives, h_primitivesCfg.planePrimitiveCount, &d_planePrimitives, d_primitivesCfg, &(d_primitivesCfg->planePrimitives));
    }

    // Copy the scalar values from host to device for plane primitives count
    cudaMemcpy(&(d_primitivesCfg->planePrimitiveCount), &(h_primitivesCfg.planePrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Quad primitives
    if (h_primitivesCfg.quadPrimitiveCount > 0)
    {
        quadPrimitiveConfig* d_quadPrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.quadPrimitives, h_primitivesCfg.quadPrimitiveCount, &d_quadPrimitives, d_primitivesCfg, &(d_primitivesCfg->quadPrimitives));
    }

    // Copy the scalar values from host to device for quad primitives count
    cudaMemcpy(&(d_primitivesCfg->quadPrimitiveCount), &(h_primitivesCfg.quadPrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Box primitives
    if (h_primitivesCfg.boxPrimitiveCount > 0)
    {
        boxPrimitiveConfig* d_boxPrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.boxPrimitives, h_primitivesCfg.boxPrimitiveCount, &d_boxPrimitives, d_primitivesCfg, &(d_primitivesCfg->boxPrimitives));
    }

    // Copy the scalar values from host to device for box primitives count
    cudaMemcpy(&(d_primitivesCfg->boxPrimitiveCount), &(h_primitivesCfg.boxPrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Cone primitives
    if (h_primitivesCfg.conePrimitiveCount > 0)
    {
        conePrimitiveConfig* d_conePrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.conePrimitives, h_primitivesCfg.conePrimitiveCount, &d_conePrimitives, d_primitivesCfg, &(d_primitivesCfg->conePrimitives));
    }

    // Copy the scalar values from host to device for cone primitives count
    cudaMemcpy(&(d_primitivesCfg->conePrimitiveCount), &(h_primitivesCfg.conePrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Cylinder primitives
    if (h_primitivesCfg.cylinderPrimitiveCount > 0)
    {
        cylinderPrimitiveConfig* d_cylinderPrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.cylinderPrimitives, h_primitivesCfg.cylinderPrimitiveCount, &d_cylinderPrimitives, d_primitivesCfg, &(d_primitivesCfg->cylinderPrimitives));
    }

    // Copy the scalar values from host to device for cylinder primitives count
    cudaMemcpy(&(d_primitivesCfg->cylinderPrimitiveCount), &(h_primitivesCfg.cylinderPrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Disk primitives
    if (h_primitivesCfg.diskPrimitiveCount > 0)
    {
        diskPrimitiveConfig* d_diskPrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.diskPrimitives, h_primitivesCfg.diskPrimitiveCount, &d_diskPrimitives, d_primitivesCfg, &(d_primitivesCfg->diskPrimitives));
    }

    // Copy the scalar values from host to device for disk primitives count
    cudaMemcpy(&(d_primitivesCfg->diskPrimitiveCount), &(h_primitivesCfg.diskPrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Torus primitives
    if (h_primitivesCfg.torusPrimitiveCount > 0)
    {
        torusPrimitiveConfig* d_torusPrimitives;
        copyCommonPrimitiveConfig(h_primitivesCfg.torusPrimitives, h_primitivesCfg.torusPrimitiveCount, &d_torusPrimitives, d_primitivesCfg, &(d_primitivesCfg->torusPrimitives));
    }

    // Copy the scalar values from host to device for torus primitives count
    cudaMemcpy(&(d_primitivesCfg->torusPrimitiveCount), &(h_primitivesCfg.torusPrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);


    // Volume primitives
    if (h_primitivesCfg.volumePrimitiveCount > 0)
    {
        volumePrimitiveConfig* d_volumePrimitives;
        copyVolumePrimitiveConfig(h_primitivesCfg.volumePrimitives, h_primitivesCfg.volumePrimitiveCount, &d_volumePrimitives, d_primitivesCfg, &(d_primitivesCfg->volumePrimitives));
    }

    // Copy the scalar values from host to device for volume primitives count
    cudaMemcpy(&(d_primitivesCfg->volumePrimitiveCount), &(h_primitivesCfg.volumePrimitiveCount), sizeof(int8_t), cudaMemcpyHostToDevice);

    return d_primitivesCfg;
}



// Main function to prepare camera
cameraConfig* prepareCamera(const cameraConfig& h_cameraCfg)
{
    cameraConfig* d_cameraCfg;
    cudaMalloc((void**)&d_cameraCfg, sizeof(cameraConfig));

    cudaMemcpy(&(d_cameraCfg->isOrthographic), &(h_cameraCfg.isOrthographic), sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->lookAt), &(h_cameraCfg.lookAt), sizeof(point3), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->lookFrom), &(h_cameraCfg.lookFrom), sizeof(point3), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->upAxis), &(h_cameraCfg.upAxis), sizeof(point3), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->aspectRatio), &(h_cameraCfg.aspectRatio), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->focus), &(h_cameraCfg.focus), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->fov), &(h_cameraCfg.fov), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->aperture), &(h_cameraCfg.aperture), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->openingTime), &(h_cameraCfg.openingTime), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_cameraCfg->orthoHeight), &(h_cameraCfg.orthoHeight), sizeof(float), cudaMemcpyHostToDevice);
    
    return d_cameraCfg;
}



// Main function to prepare output image
imageConfig* prepareImage(const imageConfig& h_imageCfg)
{
    imageConfig* d_imageCfg;
    cudaMalloc((void**)&d_imageCfg, sizeof(imageConfig));

    cudaMemcpy(&(d_imageCfg->background), &(h_imageCfg.background), sizeof(imageBackgroundConfig), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageCfg->width), &(h_imageCfg.width), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageCfg->height), &(h_imageCfg.height), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageCfg->outputFilePath), &(h_imageCfg.outputFilePath), sizeof(const char*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageCfg->spp), &(h_imageCfg.spp), sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_imageCfg->depth), &(h_imageCfg.depth), sizeof(int), cudaMemcpyHostToDevice);

    return d_imageCfg;
}


bitmap_image** load_images(const sceneConfig& sceneCfg, int width, int height, int h_images_count)
{
    int bytes_per_pixel = 3;

    // Allocate space for images pointers on the device
    bitmap_image** d_images;
    checkCudaErrors(cudaMalloc((void**)&d_images, h_images_count * sizeof(bitmap_image*)));

    // Loop over all images and load them
    for (int i = 0; i < sceneCfg.texturesCfg.imageTextureCount; ++i)
    {
        int tex_x, tex_y, tex_n;
        imageTextureConfig imageTexture = sceneCfg.texturesCfg.imageTextures[i];
        // TO DO : add bump, normal, displace textures

        unsigned char* tex_data_host = stbi_load(imageTexture.filepath, &tex_x, &tex_y, &tex_n, bytes_per_pixel);
        if (tex_data_host == nullptr)
        {
            printf("[ERROR] Failed to load texture: %s\n", imageTexture.filepath);
            continue;
        }

        // Allocate managed memory for texture data on device
        unsigned char* tex_data;
        checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Initialize texture on device
        bitmap_image** d_image;
        checkCudaErrors(cudaMalloc((void**)&d_image, sizeof(bitmap_image*)));
        image_init<<<1, 1>>> (tex_data, tex_x, tex_y, tex_n, d_image);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Store the pointer to the current texture in the array
        checkCudaErrors(cudaMemcpy(&d_images[i], d_image, sizeof(bitmap_image*), cudaMemcpyDeviceToDevice));

        // Free the host-side texture after copying to the device
        stbi_image_free(tex_data_host);
    }

    return d_images;
}

// Function to load OBJ file using tinyobjloader
// Function to load a single OBJ file using tinyobjloader
bool loadOBJ(const std::string& inputfile, MeshData& meshData)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // Load the .obj file
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty()) {
        std::cout << "Warning: " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << "Error: " << err << std::endl;
        return false;
    }
    if (!ret) {
        return false;
    }

    // Extract vertices and indices
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<int> indices;

    for (size_t s = 0; s < shapes.size(); s++) {
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];

            for (size_t v = 0; v < fv; v++) {
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);

                if (idx.normal_index >= 0) {
                    normals.push_back(attrib.normals[3 * idx.normal_index + 0]);
                    normals.push_back(attrib.normals[3 * idx.normal_index + 1]);
                    normals.push_back(attrib.normals[3 * idx.normal_index + 2]);
                }

                indices.push_back(static_cast<int>(index_offset + v));
            }
            index_offset += fv;
        }
    }

    // Allocate and copy data to the device (GPU)
    meshData.num_vertices = vertices.size();
    meshData.num_indices = indices.size();

    cudaMalloc(&meshData.d_vertices, vertices.size() * sizeof(float));
    cudaMalloc(&meshData.d_normals, normals.size() * sizeof(float));
    cudaMalloc(&meshData.d_indices, indices.size() * sizeof(int));

    cudaMemcpy(meshData.d_vertices, vertices.data(), vertices.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(meshData.d_normals, normals.data(), normals.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(meshData.d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice);

    return true;
}

// Function to load multiple OBJ files
bool loadMultipleOBJs(const std::vector<std::string>& files, SceneData& sceneData)
{
    for (const auto& file : files) {
        MeshData meshData;
        if (!loadOBJ(file, meshData)) {
            std::cerr << "Failed to load OBJ file: " << file << std::endl;
            return false;
        }
        sceneData.meshes.push_back(meshData);  // Store each mesh
    }
    return true;
}

// CUDA Kernel to process vertices of multiple meshes
__global__ void processMultipleMeshes(const MeshData* meshes, size_t num_meshes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t m = 0; m < num_meshes; ++m) {
        const MeshData& mesh = meshes[m];
        if (idx < mesh.num_vertices / 3) {  // Each vertex has 3 components (x, y, z)
            float x = mesh.d_vertices[3 * idx];
            float y = mesh.d_vertices[3 * idx + 1];
            float z = mesh.d_vertices[3 * idx + 2];

            printf("Mesh %zu - Vertex %d: x = %f, y = %f, z = %f\n", m, idx, x, y, z);
        }
    }
}


void renderGPU(const sceneConfig& sceneCfg, const cudaDeviceProp& prop, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath)
{
    std::cout << "[INFO] Rendering " << width << "x" << height << " " << spp << " samples > " << filepath << std::endl;

    setupCuda(prop);

    float ratio = (float)height / (float)width;
    int sqrt_spp = static_cast<int>(sqrt(spp));
    

    dim3 single_block(1, 1);
    dim3 single_thread(1, 1);



    // Allocating CUDA memory for render ouput
    color* d_output;
    checkCudaErrors(cudaMallocManaged((void**)&d_output, width * height * sizeof(color)));


    // Allocating CUDA memory for images
    int h_images_count = sceneCfg.texturesCfg.imageTextureCount;
    bitmap_image** h_images = load_images(sceneCfg, width, height, h_images_count);


    // Allocating CUDA memory for meshes
    SceneData sceneData;

    // List of OBJ files to load
    std::vector<std::string> objFiles = { 
        "E:\\MyProjects\\MyOwnRaytracerCuda2\\data\\models\\crate.obj", 
        "E:\\MyProjects\\MyOwnRaytracerCuda2\\data\\models\\cushion.obj" };

    if (!loadMultipleOBJs(objFiles, sceneData)) {
        std::cerr << "Failed to load one or more OBJ files!" << std::endl;
        return;
    }

    // Allocate device memory for MeshData array
    MeshData* d_meshes;
    cudaMalloc(&d_meshes, sceneData.meshes.size() * sizeof(MeshData));

    // Copy MeshData from host to device
    cudaMemcpy(d_meshes, sceneData.meshes.data(), sceneData.meshes.size() * sizeof(MeshData), cudaMemcpyHostToDevice);

    // Set up kernel launch parameters
    int blockSize = 256;
    int numBlocks = (sceneData.meshes[0].num_vertices / 3 + blockSize - 1) / blockSize;

    // Launch the kernel
    processMultipleMeshes<<<numBlocks, blockSize>>>(d_meshes, sceneData.meshes.size());

    // Wait for GPU to finish
    cudaDeviceSynchronize();





    sceneConfig* d_sceneCfg;

    // Allocate memory on the device for the top-level `sceneConfig` struct
    checkCudaErrors(cudaMalloc((void**)&d_sceneCfg, sizeof(sceneConfig)));

    texturesConfig* d_texturesCfg = prepareTextures(sceneCfg.texturesCfg);
    materialsConfig* d_materialsCfg = prepareMaterials(sceneCfg.materialsCfg);
    lightsConfig* d_lightsCfg = prepareLights(sceneCfg.lightsCfg);
    primitivesConfig* d_primitivesCfg = preparePrimitives(sceneCfg.primitivesCfg);
    cameraConfig* d_cameraCfg = prepareCamera(sceneCfg.cameraCfg);
    imageConfig* d_imageCfg = prepareImage(sceneCfg.imageCfg);

    // Now copy the lightsConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->lightsCfg, d_lightsCfg, sizeof(lightsConfig), cudaMemcpyHostToDevice));

    // Now copy the texturesConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->texturesCfg, d_texturesCfg, sizeof(texturesConfig), cudaMemcpyHostToDevice));

    // Now copy the materialsConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->materialsCfg, d_materialsCfg, sizeof(materialsConfig), cudaMemcpyHostToDevice));

    // Now copy the primitivesConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->primitivesCfg, d_primitivesCfg, sizeof(primitivesConfig), cudaMemcpyHostToDevice));

    // Now copy the cameraConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->cameraCfg, d_cameraCfg, sizeof(cameraConfig), cudaMemcpyHostToDevice));

    // Now copy the imageConfig pointer from host to device sceneConfig
    checkCudaErrors(cudaMemcpy(&d_sceneCfg->imageCfg, d_imageCfg, sizeof(imageConfig), cudaMemcpyHostToDevice));




    // World
    hittable_list** elist;
    checkCudaErrors(cudaMalloc((void**)&elist, sizeof(hittable_list*)));

    hittable_list** elights;
    checkCudaErrors(cudaMalloc((void**)&elights, sizeof(hittable_list*)));
    
    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    sampler** aa_sampler;
    checkCudaErrors(cudaMalloc((void**)&aa_sampler, sizeof(sampler*)));





    load_scene<<<single_block, single_thread>>>(d_sceneCfg, elist, elights, cam, aa_sampler, width, height, ratio, spp, sqrt_spp, h_images, h_images_count, 1984);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 render_blocks(width / tx+1, height / ty+1);
    dim3 render_threads(tx, ty);


    printf("[INFO] Render with %u/%u blocks of %u/%u threads\n", render_blocks.x, render_blocks.y, render_threads.x, render_threads.y);



    render<<<render_blocks, render_threads>>>(d_output, width, height, spp, sqrt_spp, max_depth, elist, elights, cam, aa_sampler, 2580);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


 

    // little padding to avoid remaining black zone at the end of the render preview
    for (int jj = 0; jj < 4; jj++)
    {
        for (int ii = 0; ii < width; ii++)
        {
            printf("p %u %u %u %u %u\n", ii, height - jj - 1, 0, 0, 0);
        }
    }
    


    uint8_t* imageHost = new uint8_t[width * height * 3 * sizeof(uint8_t)];
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;

            imageHost[(height - j - 1) * width * 3 + i * 3] = (uint8_t)d_output[pixel_index].r();
            imageHost[(height - j - 1) * width * 3 + i * 3 + 1] = (uint8_t)d_output[pixel_index].g();
            imageHost[(height - j - 1) * width * 3 + i * 3 + 2] = (uint8_t)d_output[pixel_index].b();
        }
    }

    // save image
    stbi_write_png(filepath, width, height, 3, imageHost, width * 3);



    // Free GPU memory
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(elights));
    checkCudaErrors(cudaFree(elist));
    checkCudaErrors(cudaFree(aa_sampler));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_sceneCfg));

    
    // Free GPU memory for each mesh
    for (const auto& mesh : sceneData.meshes) {
        cudaFree(mesh.d_vertices);
        cudaFree(mesh.d_normals);
        cudaFree(mesh.d_indices);
    }

    // Free the device MeshData array
    cudaFree(d_meshes);
}


void launchGPU(const sceneConfig& sceneCfg, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath, bool quietMode)
{
    cudaDeviceProp prop;

    if (!isGpuAvailable(prop))
    {
        return;
    }

    //std::cout << "Rendering222 " << nx << "x" << ny << " " << ns << " samples > " << filepath << std::endl;

    //std::cout << "[INFO] Use GPU device " << deviceIndex << " " << deviceName << std::endl;

    // https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/
    // __global__ - Runs on the GPU, called from the CPU or the GPU*. Executed with <<<dim3>>> arguments.
    // __device__ - Runs on the GPU, called from the GPU. Can be used with variabiles too.
    // __host__ - Runs on the CPU, called from the CPU.
    // 
    // --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --std c++20 --verbose
    // --expt-relaxed-constexpr --std c++20 -Xcudafe="--diag_suppress=20012 --diag_suppress=20208" 
    //
    renderGPU(sceneCfg, prop, width, height, spp, max_depth, tx, ty, filepath);
}
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

__device__ texture* fetchTexture(sceneConfig* sceneCfg, bitmap_image** images, const char* textureName)
{
    for (int i = 0; i < sceneCfg->texturesCfg.solidColorTextureCount; i++)
    {
        solidColorTextureConfig solidColorTexture = sceneCfg->texturesCfg.solidColorTextures[i];
        printf("[GPU] solidColorTexture%d %s %g/%g/%g\n", i,
            solidColorTexture.name,
            solidColorTexture.rgb.r(), solidColorTexture.rgb.g(), solidColorTexture.rgb.b());

        if (strcmp_device(solidColorTexture.name, textureName) == 0)
        {
            return new solid_color_texture(solidColorTexture.rgb);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.gradientColorTextureCount; i++)
    {
        gradientColorTextureConfig gradientColorTexture = sceneCfg->texturesCfg.gradientColorTextures[i];
        printf("[GPU] gradientColorTexture%d %s %g/%g/%g %g/%g/%g %d %d\n", i,
            gradientColorTexture.name,
            gradientColorTexture.color1.r(), gradientColorTexture.color1.g(), gradientColorTexture.color1.b(),
            gradientColorTexture.color2.r(), gradientColorTexture.color2.g(), gradientColorTexture.color2.b(),
            gradientColorTexture.vertical,
            gradientColorTexture.hsv);

        if (strcmp_device(gradientColorTexture.name, textureName) == 0)
        {
            return new gradient_texture(gradientColorTexture.color1, gradientColorTexture.color2, gradientColorTexture.vertical, gradientColorTexture.hsv);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.imageTextureCount; i++)
    {
        imageTextureConfig imageTexture = sceneCfg->texturesCfg.imageTextures[i];
        printf("[GPU] imageTexture%d %s %s\n", i,
            imageTexture.name,
            imageTexture.filepath);

        if (strcmp_device(imageTexture.name, textureName) == 0)
        {
            bitmap_image img = *(images[0]);
            return new image_texture(img);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.checkerTextureCount; i++)
    {
        checkerTextureConfig checkerTexture = sceneCfg->texturesCfg.checkerTextures[i];
        printf("[GPU] checkerTexture%d %s %g/%g/%g %g/%g/%g %s %s %g\n", i,
            checkerTexture.name,
            checkerTexture.oddColor.r(), checkerTexture.oddColor.g(), checkerTexture.oddColor.b(),
            checkerTexture.evenColor.r(), checkerTexture.evenColor.g(), checkerTexture.evenColor.b(),
            checkerTexture.oddTextureName,
            checkerTexture.evenTextureName,
            checkerTexture.scale);

        if (strcmp_device(checkerTexture.name, textureName) == 0)
        {
            return new checker_texture(checkerTexture.scale, checkerTexture.oddColor, checkerTexture.evenColor);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.noiseTextureCount; i++)
    {
        noiseTextureConfig noiseTexture = sceneCfg->texturesCfg.noiseTextures[i];
        printf("[GPU] noiseTexture%d %s %g\n", i,
            noiseTexture.name,
            noiseTexture.scale);

        if (strcmp_device(noiseTexture.name, textureName) == 0)
        {
            return new perlin_noise_texture(noiseTexture.scale);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.bumpTextureCount; i++)
    {
        bumpTextureConfig bumpTexture = sceneCfg->texturesCfg.bumpTextures[i];
        printf("[GPU] bumpTexture%d %s %s %g\n", i,
            bumpTexture.name,
            bumpTexture.filepath,
            bumpTexture.strength);

        if (strcmp_device(bumpTexture.name, textureName) == 0)
        {
            //return new bump_texture(bumpTexture.filepath);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.normalTextureCount; i++)
    {
        normalTextureConfig normalTexture = sceneCfg->texturesCfg.normalTextures[i];
        printf("[GPU] normalTexture%d %s %s %g\n", i,
            normalTexture.name,
            normalTexture.filepath,
            normalTexture.strength);

        if (strcmp_device(normalTexture.name, textureName) == 0)
        {
            //return new normal_texture(bumpTexture.filepath);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.displacementTextureCount; i++)
    {
        displacementTextureConfig displacementTexture = sceneCfg->texturesCfg.displacementTextures[i];
        printf("[GPU] displacementTexture%d %s %s %g\n", i,
            displacementTexture.name,
            displacementTexture.filepath,
            displacementTexture.strength);

        if (strcmp_device(displacementTexture.name, textureName) == 0)
        {
            //return new normal_texture(bumpTexture.filepath);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.alphaTextureCount; i++)
    {
        alphaTextureConfig alphaTexture = sceneCfg->texturesCfg.alphaTextures[i];
        printf("[GPU] alphaTexture%d %s %s %d\n", i,
            alphaTexture.name,
            alphaTexture.filepath,
            alphaTexture.doubleSided);

        if (strcmp_device(alphaTexture.name, textureName) == 0)
        {
            //return new alpha_texture(bumpTexture.filepath);
        }
    }

    for (int i = 0; i < sceneCfg->texturesCfg.emissiveTextureCount; i++)
    {
        emissiveTextureConfig emissiveTexture = sceneCfg->texturesCfg.emissiveTextures[i];
        printf("[GPU] emissiveTexture%d %s %s %g\n", i,
            emissiveTexture.name,
            emissiveTexture.filepath,
            emissiveTexture.strength);

        if (strcmp_device(emissiveTexture.name, textureName) == 0)
        {
            //return new emissive_texture(bumpTexture.filepath);
        }
    }
}


__device__ material* fetchMaterial(sceneConfig* sceneCfg, bitmap_image** images, const char* materialName)
{
    for (int i = 0; i < sceneCfg->materialsCfg.lambertianMaterialCount; i++)
    {
        lambertianMaterialConfig lambertianMaterial = sceneCfg->materialsCfg.lambertianMaterials[i];
        printf("[GPU] lambertianMaterial%d %s %g/%g/%g %s\n", i,
            lambertianMaterial.name,
            lambertianMaterial.rgb.r(), lambertianMaterial.rgb.g(), lambertianMaterial.rgb.b(),
            lambertianMaterial.textureName);

        if (strcmp_device(lambertianMaterial.name, materialName) == 0)
        {
            if (lambertianMaterial.textureName != nullptr && lambertianMaterial.textureName[0] != '\0')
            {
                texture* tex = fetchTexture(sceneCfg, images, lambertianMaterial.textureName);
                if (tex)
                {
                    return new lambertian(tex);
                }
            }
            else
            {
                return new lambertian(lambertianMaterial.rgb);
            }
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.metalMaterialCount; i++)
    {
        metalMaterialConfig metalMaterial = sceneCfg->materialsCfg.metalMaterials[i];
        printf("[GPU] metalMaterial%d %s %g/%g/%g %g\n", i,
            metalMaterial.name,
            metalMaterial.rgb.r(), metalMaterial.rgb.g(), metalMaterial.rgb.b(),
            metalMaterial.fuzziness);

        if (strcmp_device(metalMaterial.name, materialName) == 0)
        {
            return new metal(metalMaterial.rgb, metalMaterial.fuzziness);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.dielectricMaterialCount; i++)
    {
        dielectricMaterialConfig glassMaterial = sceneCfg->materialsCfg.dielectricMaterials[i];
        printf("[GPU] glassMaterial%d %s %g\n", i,
            glassMaterial.name,
            glassMaterial.refraction);

        if (strcmp_device(glassMaterial.name, materialName) == 0)
        {
            return new dielectric(glassMaterial.refraction);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.isotropicMaterialCount; i++)
    {
        isotropicMaterialConfig isotropicMaterial = sceneCfg->materialsCfg.isotropicMaterials[i];
        printf("[GPU] isotropicMaterial%d %s %g/%g/%g %s\n", i,
            isotropicMaterial.name,
            isotropicMaterial.rgb.r(), isotropicMaterial.rgb.g(), isotropicMaterial.rgb.b(),
            isotropicMaterial.textureName);

        if (strcmp_device(isotropicMaterial.name, materialName) == 0)
        {
            if (isotropicMaterial.textureName != nullptr && isotropicMaterial.textureName[0] != '\0')
                return new isotropic(isotropicMaterial.rgb);
            else
                return new isotropic(isotropicMaterial.rgb);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.anisotropicMaterialCount; i++)
    {
        anisotropicMaterialConfig anisotropicMaterial = sceneCfg->materialsCfg.anisotropicMaterials[i];
        printf("[GPU] anisotropicMaterial%d %s %g/%g/%g %g %g %s %s %s %g\n", i,
            anisotropicMaterial.name,
            anisotropicMaterial.rgb.r(), anisotropicMaterial.rgb.g(), anisotropicMaterial.rgb.b(),
            anisotropicMaterial.nuf,
            anisotropicMaterial.nvf,
            anisotropicMaterial.diffuseTextureName,
            anisotropicMaterial.specularTextureName,
            anisotropicMaterial.exponentTextureName,
            anisotropicMaterial.roughness);

        if (strcmp_device(anisotropicMaterial.name, materialName) == 0)
        {
            //return new anisotropic(anisotropicMaterial.nuf, anisotropicMaterial.nvf);
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.orenNayarMaterialCount; i++)
    {
        orenNayarMaterialConfig orenNayarMaterial = sceneCfg->materialsCfg.orenNayarMaterials[i];
        printf("[GPU] orenNayarMaterial%d %s %g/%g/%g %s %g %g\n", i,
            orenNayarMaterial.name,
            orenNayarMaterial.rgb.r(), orenNayarMaterial.rgb.g(), orenNayarMaterial.rgb.b(),
            orenNayarMaterial.textureName,
            orenNayarMaterial.roughness,
            orenNayarMaterial.albedo_temp);

        if (strcmp_device(orenNayarMaterial.name, materialName) == 0)
        {
            //return oren_nayar();
        }
    }

    for (int i = 0; i < sceneCfg->materialsCfg.phongMaterialCount; i++)
    {
        phongMaterialConfig phongMaterial = sceneCfg->materialsCfg.phongMaterials[i];
        printf("[GPU] phongMaterial%d %s %s %s %s %s %s %s %s %g/%g/%g\n", i,
            phongMaterial.name,
            phongMaterial.diffuseTextureName,
            phongMaterial.specularTextureName,
            phongMaterial.bumpTextureName,
            phongMaterial.normalTextureName,
            phongMaterial.displacementTextureName,
            phongMaterial.alphaTextureName,
            phongMaterial.emissiveTextureName,
            phongMaterial.ambientColor.r(), phongMaterial.ambientColor.g(), phongMaterial.ambientColor.b());

        if (strcmp_device(phongMaterial.name, materialName) == 0)
        {
            //return new phong();
        }
    }
}

__global__ void load_scene(sceneConfig* sceneCfg, hittable_list **elist, hittable_list **elights,  camera **cam, sampler **aa_sampler, int width, int height, float ratio, int spp, int sqrt_spp, bitmap_image** images, int num_textures, int seed)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // thrust random engine and distribution
        thrust::minstd_rand rng(seed);
        thrust::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);


        *elights = new hittable_list();

        *elist = new hittable_list();




        // TEXTURES DATA
        // You can access individual textures like this:
        // textures[0] -> first texture
        // textures[1] -> second texture
        // num_textures -> number of textures
        // Example usage:
        //for (int i = 0; i < num_textures; ++i)
        //{
        //    // Process each texture as needed
        //    bitmap_image* current_texture = images[i];
        //    // Use current_texture in the scene (e.g., for material or texture mapping)
        //}



        // LIGHTS
        printf("[GPU] %i omni lights found\n", sceneCfg->lightsCfg.omniLightCount);
        printf("[GPU] %i dir lights found\n", sceneCfg->lightsCfg.dirLightCount);
        printf("[GPU] %i spot lights found\n", sceneCfg->lightsCfg.spotLightCount);

        for (int i = 0; i < sceneCfg->lightsCfg.omniLightCount; i++)
        {
            omniLightConfig omnilight = sceneCfg->lightsCfg.omniLights[i];
            printf("[GPU] omnilight%d %g %s %g/%g/%g %g/%g/%g %g %d\n", i, 
                omnilight.intensity, omnilight.name,
                omnilight.position.x, omnilight.position.y, omnilight.position.z,
                omnilight.rgb.r(), omnilight.rgb.g(), omnilight.rgb.b(),
                omnilight.radius,
                omnilight.invisible);
        }

        for (int i = 0; i < sceneCfg->lightsCfg.dirLightCount; i++)
        {
            directionalLightConfig dirlight = sceneCfg->lightsCfg.dirLights[i];
            printf("[GPU] dirlight%d %g %s %g/%g/%g %g/%g/%g %g/%g/%g %g/%g/%g %d\n", i, 
                dirlight.intensity, dirlight.name,
                dirlight.position.x, dirlight.position.y, dirlight.position.z,
                dirlight.u.x, dirlight.u.y, dirlight.u.z,
                dirlight.v.x, dirlight.v.y, dirlight.v.z,
                dirlight.rgb.r(), dirlight.rgb.g(), dirlight.rgb.b(),
                dirlight.invisible);
        }

        for (int i = 0; i < sceneCfg->lightsCfg.spotLightCount; i++)
        {
            spotLightConfig spotlight = sceneCfg->lightsCfg.spotLights[i];
            printf("[GPU] spotlight%d %g %s %g/%g/%g %g/%g/%g %g %g %g %g/%g/%g %d\n", i, 
                spotlight.intensity, spotlight.name,
                spotlight.position.x, spotlight.position.y, spotlight.position.z,
                spotlight.direction.x, spotlight.direction.y, spotlight.direction.z,
                spotlight.cutoff,
                spotlight.falloff,
                spotlight.radius,
                spotlight.rgb.r(), spotlight.rgb.g(), spotlight.rgb.b(),
                spotlight.invisible);
        }


        // PRIMITIVES
        for (int i = 0; i < sceneCfg->primitivesCfg.spherePrimitiveCount; i++)
        {
            spherePrimitiveConfig spherePrimitive = sceneCfg->primitivesCfg.spherePrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, spherePrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createSphere(spherePrimitive.name, spherePrimitive.position, spherePrimitive.radius, mat, spherePrimitive.mapping));

            printf("[GPU] spherePrimitive%d %s %g/%g/%g %g/%g %g/%g %g/%g %g %s %s\n", i,
                spherePrimitive.name,
                spherePrimitive.position.x, spherePrimitive.position.y, spherePrimitive.position.z,
                spherePrimitive.mapping.offset_u(), spherePrimitive.mapping.offset_v(),
                spherePrimitive.mapping.repeat_u(), spherePrimitive.mapping.repeat_v(),
                spherePrimitive.mapping.scale_u(), spherePrimitive.mapping.scale_v(),
                spherePrimitive.radius,
                spherePrimitive.materialName,
                spherePrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.planePrimitiveCount; i++)
        {
            planePrimitiveConfig planePrimitive = sceneCfg->primitivesCfg.planePrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, planePrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createPlane(planePrimitive.name, planePrimitive.point1, planePrimitive.point2, mat, planePrimitive.mapping));

            printf("[GPU] planePrimitive%d %s %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g %s %s\n", i,
                planePrimitive.name,
                planePrimitive.point1.x, planePrimitive.point1.y, planePrimitive.point1.z,
                planePrimitive.point2.x, planePrimitive.point2.y, planePrimitive.point2.z,
                planePrimitive.mapping.offset_u(), planePrimitive.mapping.offset_v(),
                planePrimitive.mapping.repeat_u(), planePrimitive.mapping.repeat_v(),
                planePrimitive.mapping.scale_u(), planePrimitive.mapping.scale_v(),
                planePrimitive.materialName,
                planePrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.quadPrimitiveCount; i++)
        {
            quadPrimitiveConfig quadPrimitive = sceneCfg->primitivesCfg.quadPrimitives[i];
            
            material* mat = fetchMaterial(sceneCfg, images, quadPrimitive.materialName);
            
            if (mat)
                (*elist)->add(scene_factory::createQuad(quadPrimitive.name, quadPrimitive.position, quadPrimitive.u, quadPrimitive.v, mat, quadPrimitive.mapping));

            printf("[GPU] quadPrimitive%d %s %g/%g/%g %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g %s %s\n", i,
                quadPrimitive.name,
                quadPrimitive.position.x, quadPrimitive.position.y, quadPrimitive.position.z,
                quadPrimitive.u.x, quadPrimitive.u.y, quadPrimitive.u.z,
                quadPrimitive.v.x, quadPrimitive.v.y, quadPrimitive.v.z,
                quadPrimitive.mapping.offset_u(), quadPrimitive.mapping.offset_v(),
                quadPrimitive.mapping.repeat_u(), quadPrimitive.mapping.repeat_v(),
                quadPrimitive.mapping.scale_u(), quadPrimitive.mapping.scale_v(),
                quadPrimitive.materialName,
                quadPrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.boxPrimitiveCount; i++)
        {
            boxPrimitiveConfig boxPrimitive = sceneCfg->primitivesCfg.boxPrimitives[i];

            material* mat = fetchMaterial(sceneCfg, images, boxPrimitive.materialName);

            if (mat)
                (*elist)->add(scene_factory::createBox(boxPrimitive.name, boxPrimitive.position, boxPrimitive.size, mat, boxPrimitive.mapping));


            printf("[GPU] boxPrimitive%d %s %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g %s %s\n", i,
                boxPrimitive.name,
                boxPrimitive.position.x, boxPrimitive.position.y, boxPrimitive.position.z,
                boxPrimitive.size.x, boxPrimitive.size.y, boxPrimitive.size.z,
                boxPrimitive.mapping.offset_u(), boxPrimitive.mapping.offset_v(),
                boxPrimitive.mapping.repeat_u(), boxPrimitive.mapping.repeat_v(),
                boxPrimitive.mapping.scale_u(), boxPrimitive.mapping.scale_v(),
                boxPrimitive.materialName,
                boxPrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.conePrimitiveCount; i++)
        {
            conePrimitiveConfig conePrimitive = sceneCfg->primitivesCfg.conePrimitives[i];
            printf("[GPU] conePrimitive%d %s %g/%g/%g %g %g %g/%g %g/%g %g/%g %s %s\n", i,
                conePrimitive.name,
                conePrimitive.position.x, conePrimitive.position.y, conePrimitive.position.z,
                conePrimitive.radius,
                conePrimitive.height,
                conePrimitive.mapping.offset_u(), conePrimitive.mapping.offset_v(),
                conePrimitive.mapping.repeat_u(), conePrimitive.mapping.repeat_v(),
                conePrimitive.mapping.scale_u(), conePrimitive.mapping.scale_v(),
                conePrimitive.materialName,
                conePrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.cylinderPrimitiveCount; i++)
        {
            cylinderPrimitiveConfig cylinderPrimitive = sceneCfg->primitivesCfg.cylinderPrimitives[i];
            printf("[GPU] cylinderPrimitive%d %s %g/%g/%g %g %g %g/%g %g/%g %g/%g %s %s\n", i,
                cylinderPrimitive.name,
                cylinderPrimitive.position.x, cylinderPrimitive.position.y, cylinderPrimitive.position.z,
                cylinderPrimitive.radius,
                cylinderPrimitive.height,
                cylinderPrimitive.mapping.offset_u(), cylinderPrimitive.mapping.offset_v(),
                cylinderPrimitive.mapping.repeat_u(), cylinderPrimitive.mapping.repeat_v(),
                cylinderPrimitive.mapping.scale_u(), cylinderPrimitive.mapping.scale_v(),
                cylinderPrimitive.materialName,
                cylinderPrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.diskPrimitiveCount; i++)
        {
            diskPrimitiveConfig diskPrimitive = sceneCfg->primitivesCfg.diskPrimitives[i];
            printf("[GPU] diskPrimitive%d %s %g/%g/%g %g %g %g/%g %g/%g %g/%g %s %s\n", i,
                diskPrimitive.name,
                diskPrimitive.position.x, diskPrimitive.position.y, diskPrimitive.position.z,
                diskPrimitive.radius,
                diskPrimitive.height,
                diskPrimitive.mapping.offset_u(), diskPrimitive.mapping.offset_v(),
                diskPrimitive.mapping.repeat_u(), diskPrimitive.mapping.repeat_v(),
                diskPrimitive.mapping.scale_u(), diskPrimitive.mapping.scale_v(),
                diskPrimitive.materialName,
                diskPrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.torusPrimitiveCount; i++)
        {
            torusPrimitiveConfig torusPrimitive = sceneCfg->primitivesCfg.torusPrimitives[i];
            printf("[GPU] diskPrimitive%d %s %g/%g/%g %g %g %g/%g %g/%g %g/%g %s %s\n", i,
                torusPrimitive.name,
                torusPrimitive.position.x, torusPrimitive.position.y, torusPrimitive.position.z,
                torusPrimitive.minor_radius,
                torusPrimitive.major_radius,
                torusPrimitive.mapping.offset_u(), torusPrimitive.mapping.offset_v(),
                torusPrimitive.mapping.repeat_u(), torusPrimitive.mapping.repeat_v(),
                torusPrimitive.mapping.scale_u(), torusPrimitive.mapping.scale_v(),
                torusPrimitive.materialName,
                torusPrimitive.groupName);
        }

        for (int i = 0; i < sceneCfg->primitivesCfg.volumePrimitiveCount; i++)
        {
            volumePrimitiveConfig volumePrimitive = sceneCfg->primitivesCfg.volumePrimitives[i];
            printf("[GPU] volumePrimitive%d %s %s %g %g/%g/%g %s %s\n", i,
                volumePrimitive.name,
                volumePrimitive.boundaryName,
                volumePrimitive.density,
                volumePrimitive.rgb.r(), volumePrimitive.rgb.g(), volumePrimitive.rgb.b(),
                volumePrimitive.textureName,
                volumePrimitive.groupName);
        }



        //hittable_list obj = composer->getObjects();
        //for (int w = 0; w < obj.object_count; w++)
        //{
        //    auto sss = obj.objects[w];
        //    (*elist)->add(sss);
        //}


        




        //(*elist)->add(new rt::flip_normals(new yz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.12, 0.45, 0.15))), "MyLeft")));
        //(*elist)->add(new yz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(color(0.65, 0.05, 0.05))), "MyRight"));
        //(*elist)->add(new xz_rect(0, 555, 0, 555, 0, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyGround"));
        //(*elist)->add(new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyTop")));
        //(*elist)->add(new rt::flip_normals(new xz_rect(0, 555, 0, 555, 555, new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyBottom")));
        
        // back
        //(*elist)->add(new quad(point3(0,0,555), vector3(555,0,0), vector3(0,555,0), new lambertian(new solid_color_texture(color(0.73, 0.73, 0.73))), "MyBack"));


        // box
        (*elist)->add(new rt::translate(new box(point3(0.0f, 0.0f, 200.0f), vector3(165, 330, 165), new lambertian(new image_texture(*(images[0]))), "MyBox"), vector3(120,0,320)));
        
        // sphere
        //(*elist)->add(new sphere(point3(350.0f, 50.0f, 295.0f), 100.0f, new lambertian(*texture), "MySphere"));


        // light
        (*elist)->add(new directional_light(point3(278, 554, 332), vector3(-305, 0, 0), vector3(0, 0, -305), 1.0f, color(10.0, 10.0, 10.0), "MyLight", true));




        // temp extract_emissive_objects
        for (int i = 0; i < (*elist)->object_count; i++)
        {
            if ((*elist)->objects[i]->getTypeID() == HittableTypeID::lightDirectionalType)
            {
                light* derived = static_cast<light*>((*elist)->objects[i]);
                if (derived)
                {
                    (*elights)->add((*elist)->objects[i]);
                }
            }
        }

        *cam = new perspective_camera();
        (*cam)->initialize(
            vector3(278, 278, -800),
            vector3(278, 278, 0),
            vector3(0, 1, 0),
            width,
            ratio,
            40.0f,
            0.0f,
            10.0f,
            0.0f,
            0.0f,
            1.0f,
            sqrt_spp);


        // calculate bounding boxes to speed up ray computing
        *elist = new hittable_list(new bvh_node((*elist)->objects, 0, (*elist)->object_count, rng));
    }
}

__global__ void texture_init(unsigned char* tex_data, int width, int height, int channels, bitmap_image** tex)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *tex = new bitmap_image(tex_data, width, height, channels);
    }
}

__global__ void render(scene* world_scene, color* fb, int width, int height, int spp, int sqrt_spp, int max_depth, hittable_list** world, hittable_list** lights, camera** cam, sampler** aa_sampler, int seed)
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



    const size_t newPrintfFifoSize = 10000000;

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
/// // Helper function to copy texture configuration
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

void renderGPU(const sceneConfig& sceneCfg, const cudaDeviceProp& prop, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath)
{
    std::cout << "[INFO] Rendering " << width << "x" << height << " " << spp << " samples > " << filepath << std::endl;

    setupCuda(prop);



    float ratio = (float)height / (float)width;
    int sqrt_spp = static_cast<int>(sqrt(spp));
    

    dim3 single_block(1, 1);
    dim3 single_thread(1, 1);


    //int bytes_per_pixel = 3;
    //int tex_x, tex_y, tex_n;
    //unsigned char *tex_data_host = stbi_load("e:\\uv_mapper_no_numbers.jpg", &tex_x, &tex_y, &tex_n, bytes_per_pixel);
    //if (!tex_data_host) {
    //    std::cerr << "[ERROR] Failed to load texture." << std::endl;
    //    return;
    //}

    //unsigned char *tex_data;
    //checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
    //checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

    //image_texture** h_texture;
    //checkCudaErrors(cudaMalloc((void **)&h_texture, sizeof(image_texture*)));
    //texture_init<<<single_block, single_thread>>>(tex_data, tex_x, tex_y, tex_n, h_texture);




    // Allocating CUDA memory
    color* d_output;
    checkCudaErrors(cudaMallocManaged((void**)&d_output, width * height * sizeof(color)));


    scene* world_device;
    checkCudaErrors(cudaMalloc((void**)&world_device, sizeof(scene)));
    //checkCudaErrors(cudaMemcpy(world_device, &world, sizeof(scene), cudaMemcpyHostToDevice));


    int bytes_per_pixel = 3;

    // Number of images (textures)
    int num_textures = sceneCfg.texturesCfg.imageTextureCount;

    // Allocate space for texture pointers on the device
    bitmap_image** h_textures;
    checkCudaErrors(cudaMalloc((void**)&h_textures, num_textures * sizeof(bitmap_image*)));

    // Loop over all images and load them
    for (int i = 0; i < num_textures; ++i)
    {
        int tex_x, tex_y, tex_n;
        imageTextureConfig imageTexture = sceneCfg.texturesCfg.imageTextures[i];
        unsigned char* tex_data_host = stbi_load(imageTexture.filepath, &tex_x, &tex_y, &tex_n, bytes_per_pixel);
        if (!tex_data_host) {
            std::cerr << "[ERROR] Failed to load texture: " << imageTexture.filepath << std::endl;
            return;
        }

        // Allocate managed memory for texture data on device
        unsigned char* tex_data;
        checkCudaErrors(cudaMallocManaged(&tex_data, tex_x * tex_y * tex_n * sizeof(unsigned char)));
        checkCudaErrors(cudaMemcpy(tex_data, tex_data_host, tex_x * tex_y * tex_n * sizeof(unsigned char), cudaMemcpyHostToDevice));

        // Initialize texture on device
        bitmap_image** h_texture;
        checkCudaErrors(cudaMalloc((void**)&h_texture, sizeof(bitmap_image*)));
        texture_init<<<single_block, single_thread>>>(tex_data, tex_x, tex_y, tex_n, h_texture);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        // Store the pointer to the current texture in the array
        checkCudaErrors(cudaMemcpy(&h_textures[i], h_texture, sizeof(bitmap_image*), cudaMemcpyDeviceToDevice));

        // Free the host-side texture after copying to the device
        stbi_image_free(tex_data_host);
    }



    sceneConfig* d_sceneCfg;

    // Allocate memory on the device for the top-level `sceneConfig` struct
    cudaMalloc((void**)&d_sceneCfg, sizeof(sceneConfig));


    
    texturesConfig* d_texturesCfg = prepareTextures(sceneCfg.texturesCfg);
    materialsConfig* d_materialsCfg = prepareMaterials(sceneCfg.materialsCfg);
    lightsConfig* d_lightsCfg = prepareLights(sceneCfg.lightsCfg);
    primitivesConfig* d_primitivesCfg = preparePrimitives(sceneCfg.primitivesCfg);



    // Now copy the lightsConfig pointer from host to device sceneConfig
    cudaMemcpy(&d_sceneCfg->lightsCfg, d_lightsCfg, sizeof(lightsConfig), cudaMemcpyHostToDevice);

    // Now copy the texturesConfig pointer from host to device sceneConfig
    cudaMemcpy(&d_sceneCfg->texturesCfg, d_texturesCfg, sizeof(texturesConfig), cudaMemcpyHostToDevice);

    // Now copy the materialsConfig pointer from host to device sceneConfig
    cudaMemcpy(&d_sceneCfg->materialsCfg, d_materialsCfg, sizeof(materialsConfig), cudaMemcpyHostToDevice);

    // Now copy the primitivesConfig pointer from host to device sceneConfig
    cudaMemcpy(&d_sceneCfg->primitivesCfg, d_primitivesCfg, sizeof(primitivesConfig), cudaMemcpyHostToDevice);





    // World
    hittable_list **elist;
    checkCudaErrors(cudaMalloc((void**)&elist, sizeof(hittable_list*)));

    hittable_list **elights;
    checkCudaErrors(cudaMalloc((void**)&elights, sizeof(hittable_list*)));
    
    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    sampler** aa_sampler;
    checkCudaErrors(cudaMalloc((void**)&aa_sampler, sizeof(sampler*)));






    load_scene<<<single_block, single_thread>>>(d_sceneCfg, elist, elights, cam, aa_sampler, width, height, ratio, spp, sqrt_spp, h_textures, num_textures, 1984);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 render_blocks(width / tx+1, height / ty+1);
    dim3 render_threads(tx, ty);


    printf("[INFO] Render with %u/%u blocks of %u/%u threads\n", render_blocks.x, render_blocks.y, render_threads.x, render_threads.y);


    render<<<render_blocks, render_threads>>>(world_device, d_output, width, height, spp, sqrt_spp, max_depth, elist, elights, cam, aa_sampler, 2580);
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

    // Clean up
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(elights));
    checkCudaErrors(cudaFree(elist));
    checkCudaErrors(cudaFree(world_device));
    checkCudaErrors(cudaFree(aa_sampler));
    checkCudaErrors(cudaFree(d_output));
    checkCudaErrors(cudaFree(d_sceneCfg));
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


//int main(int argc, char* argv[])
//{
//    launchGPU(256, 144, 10, 2, 16, 16, "e:\\ttt2.png", true);
//}
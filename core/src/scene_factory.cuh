#pragma once





//#include "utilities/mesh_loader.cuh"




#include "primitives/box.cuh"
#include "primitives/cone.cuh"
#include "primitives/sphere.cuh"
#include "primitives/cylinder.cuh"
#include "primitives/disk.cuh"
#include "primitives/torus.cuh"
#include "primitives/aarect.cuh"
#include "primitives/quad.cuh"
#include "primitives/volume.cuh"

#include "lights/directional_light.cuh"
#include "lights/omni_light.cuh"
#include "lights/spot_light.cuh"



class scene_factory
{
public:
    __host__ __device__ scene_factory() = delete;

    

    __host__ __device__ static hittable* createBox(const char* name, const point3& position, const point3& size, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createCylinder(const char* name, const point3& position, float radius, float height, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createSphere(const char* name, const point3& position, float radius, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createCone(const char* name, const point3& position, float height, float radius, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createDisk(const char* name, const point3& position, float height, float radius, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createTorus(const char* name, const point3& position, float major_radius, float minor_radius, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createQuad(const char* name, const point3& position, const vector3 u, const vector3 v, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createPlane(const char* name, const point3& p0, const point3& p1, material* material, const uvmapping& uv, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createVolume(const char* name, hittable* boundary, float density, const color& rgb, texture* texture, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createMesh(const char* name, const point3& center, const char* filepath, material* material, const bool use_mtl, const bool use_smoothing, const rt::transform& trs, bool debug = false);

    __host__ __device__ static hittable* createDirLight(const char* name, const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible, bool debug = false);

    __host__ __device__ static hittable* createOmniLight(const char* name, const point3& pos, float radius, float intensity, color rgb, bool invisible, bool debug = false);

    __host__ __device__ static hittable* createSpotLight(const char* name, const point3& pos, const vector3& dir, float cutoff, float falloff, float intensity, float radius, color rgb, bool invisible, bool debug = false);



    __host__ __device__ static hittable* applyTransform(hittable* primitive, const rt::transform& trs);







    __host__ __device__ static material* createLambertianMaterial(const char* name, const color& rgb, bool debug = false);
    
    __host__ __device__ static material* createLambertianMaterial(const char* name, const char* textureName, texture* texture, bool debug = false);

    __host__ __device__ static material* createMetalMaterial(const char* name, const color& rgb, float fuzziness, bool debug = false);

    __host__ __device__ static material* createDielectricMaterial(const char* name, float refraction, bool debug = false);

    __host__ __device__ static material* createIsotropicMaterial(const char* name, const color& rgb, bool debug = false);

    __host__ __device__ static material* createIsotropicMaterial(const char* name, const char* textureName, texture* texture, bool debug = false);

    __host__ __device__ static material* createAnisotropicMaterial(const char* name, float nuf, float nvf, const char* diffuseTextureName, texture* diffuseTexture, const char* specularTextureName, texture* specularTexture, const char* exponentTextureName, texture* exponentTexture, bool debug = false);

    __host__ __device__ static material* createOrenNayarMaterial(const char* name, const color& rgb, float roughness, float albedo_temp, bool debug = false);

    __host__ __device__ static material* createOrenNayarMaterial(const char* name, const char* textureName, texture* texture, float roughness, float albedo_temp, bool debug = false);

    __host__ __device__ static material* createPhongMaterial(const char* name, const char* diffuseTextureName, texture* diffuseTexture, const char* specularTextureName, texture* specularTexture, const char* bumpTextureName, texture* bumpTexture, const char* normalTextureName, texture* normalTexture,
        const char* displaceTextureName, texture* displaceTexture, const char* alphaTextureName, texture* alphaTexture, const char* emissiveTextureName,
        texture* emissiveTexture, const color& ambientColor, float shininess, bool debug = false);




    __host__ __device__ static texture* createColorTexture(const char* name, const color& rgb, bool debug = false);

    __host__ __device__ static texture* createGradientTexture(const char* name, const color& color1, const color& color2, bool vertical, bool hsv, bool debug = false);

    __host__ __device__ static texture* createImageTexture(const char* name, const char* filepath, const bitmap_image& img, bool debug = false);

    __host__ __device__ static texture* createCheckerTexture(const char* name, const color& oddColor, const color& evenColor, texture* oddTexture, texture* evenTexture, float scale, bool debug = false);


    __host__ __device__ static texture* createCheckerTexture(const char* name, texture* oddTexture, const char* oddTextureName, texture* evenTexture, const char* eventTextureName, float scale, bool debug = false);

    __host__ __device__ static texture* createNoiseTexture(const char* name, float scale, bool debug = false);

    __host__ __device__ static texture* createBumpTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug = false);

    __host__ __device__ static texture* createNormalTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug = false);

    __host__ __device__ static texture* createDisplaceTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug = false);

    __host__ __device__ static texture* createAlphaTexture(const char* name, const char* filepath, const bitmap_image& img, bool doudle_sided, bool debug = false);

    __host__ __device__ static texture* createEmissiveTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug = false);
};





__host__ __device__ hittable* scene_factory::createBox(
        const char* name,
        const point3& position,
        const point3& size,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] boxPrimitive %s %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            size.x, size.y, size.z,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());

    auto primitive = new box(position, size, material, uv, name);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createCylinder(
        const char* name,
        const point3& position,
        float radius,
        float height,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] cylinderPrimitive %s %g/%g/%g %g %g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            radius,
            height,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());
    
    auto primitive = new cylinder(position, radius, height, material, uv);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createDisk(
        const char* name,
        const point3& position,
        float radius,
        float height,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] diskPrimitive %s %g/%g/%g %g %g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            radius,
            height,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());
    
    auto primitive = new disk(position, radius, height, material, uv);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createTorus(
        const char* name,
        const point3& position,
        float major_radius,
        float minor_radius,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] torusPrimitive %s %g/%g/%g %g %g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            minor_radius,
            major_radius,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());
    
    auto primitive = new torus(position, major_radius, minor_radius, material, uv);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createSphere(
        const char* name,
        const point3& position,
        float radius,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] spherePrimitive %s %g/%g/%g %g/%g %g/%g %g/%g %g\n",
            name,
            position.x, position.y, position.z,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v(),
            radius);
    
    auto primitive = new sphere(position, radius, material, uv, name);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createCone(
        const char* name,
        const point3& position,
        float height,
        float radius,
        material* material,
        const uvmapping& uv,
        const rt::transform& trs,
        bool debug)
{
    if (debug)
        printf("[GPU] conePrimitive %s %g/%g/%g %g %g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            radius,
            height,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());
    
    auto primitive = new cone(position, radius, height, material, uv, name);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createPlane(
    const char* name,
    const point3 &p0,
    const point3 &p1,
    material* material,
    const uvmapping& uv,
    const rt::transform& trs,
    bool debug)
{
    if (debug)
        printf("[GPU] planePrimitive %s %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g\n",
            name,
            p0.x, p0.y, p0.z,
            p1.x, p1.y, p1.z,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());

    
    if (p0.x == p1.x)
    {
        float x = p0.x;
        float y0 = p0.y < p1.y ? p0.y : p1.y;
        float y1 = p0.y < p1.y ? p1.y : p0.y;
        float z0 = p0.z < p1.z ? p0.z : p1.z;
        float z1 = p0.z < p1.z ? p1.z : p0.z;

        auto primitive = new yz_rect(y0, y1, z0, z1, x, material, uv, name);
        return applyTransform(primitive, trs);
    }

    if (p0.y == p1.y)
    {
        float y = p0.y;
        float x0 = p0.x < p1.x ? p0.x : p1.x;
        float x1 = p0.x < p1.x ? p1.x : p0.x;
        float z0 = p0.z < p1.z ? p0.z : p1.z;
        float z1 = p0.z < p1.z ? p1.z : p0.z;

        auto primitive = new xz_rect(x0, x1, z0, z1, y, material, uv, name);
        return applyTransform(primitive, trs);
    }

    if (p0.z == p1.z)
    {
        float z = p0.z;
        float x0 = p0.x < p1.x ? p0.x : p1.x;
        float x1 = p0.x < p1.x ? p1.x : p0.x;
        float y0 = p0.y < p1.y ? p0.y : p1.y;
        float y1 = p0.y < p1.y ? p1.y : p0.y;

        auto primitive = new xy_rect(x0, x1, y0, y1, z, material, uv, name);
        return applyTransform(primitive, trs);
    }

    return nullptr;
}

__host__ __device__ hittable* scene_factory::createQuad(
    const char* name,
    const point3& position,
    const vector3 u,
    const vector3 v,
    material* material,
    const uvmapping& uv,
    const rt::transform& trs,
    bool debug)
{
    if (debug)
        printf("[GPU] quadPrimitive %s %g/%g/%g %g/%g/%g %g/%g/%g %g/%g %g/%g %g/%g\n",
            name,
            position.x, position.y, position.z,
            u.x, u.y, u.z,
            v.x, v.y, v.z,
            uv.offset_u(), uv.offset_v(),
            uv.repeat_u(), uv.repeat_v(),
            uv.scale_u(), uv.scale_v());
    
    auto primitive = new quad(position, u, v, material, uv, name);
    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createVolume(
    const char* name,
    hittable* boundary,
    float density,
    const color& rgb,
    texture* texture,
    const rt::transform& trs,
    bool debug)
{
    if (debug)
        printf("[GPU] volumePrimitive %s %s %g %g/%g/%g\n",
            name,
            boundary->getName(),
            density,
            rgb.r(), rgb.g(), rgb.b());

    volume* primitive = nullptr;

    if (texture != nullptr)
        primitive = new volume(boundary, density, texture, name);
    else
        primitive = new volume(boundary, density, rgb, name);

    return applyTransform(primitive, trs);
}

__host__ __device__ hittable* scene_factory::createMesh(
    const char* name,
	const point3& center,
	const char* filepath,
	material* material,
	const bool use_mtl,
    const bool use_smoothing,
    const rt::transform& trs,
    bool debug)
{
    hittable* mesh = nullptr;
    
    //mesh_loader::mesh_data data;
    //
    //if (mesh_loader::load_model_from_file(filepath, data))
    //{
    //    mesh = mesh_loader::convert_model_from_file(data, material, use_mtl, use_smoothing, name);
    //}

    return mesh;
}




__host__ __device__ material* scene_factory::createLambertianMaterial(
    const char* name,
    const color& rgb,
    bool debug)
{
    if (debug)
        printf("[GPU] lambertianMaterial %s %g/%g/%g\n",
            name,
            rgb.r(), rgb.g(), rgb.b());

    return new lambertian(rgb);
}

__host__ __device__ material* scene_factory::createLambertianMaterial(
    const char* name,
    const char* textureName,
    texture* texture,
    bool debug)
{
    if (debug)
        printf("[GPU] lambertianMaterial %s %s\n",
            name,
            textureName);

    return new lambertian(texture);
}

__host__ __device__ material* scene_factory::createMetalMaterial(
    const char* name,
    const color& rgb,
    float fuzziness,
    bool debug)
{
    if (debug)
        printf("[GPU] metalMaterial %s %g/%g/%g %g\n",
            name,
            rgb.r(), rgb.g(), rgb.b(),
            fuzziness);

    return new metal(rgb, fuzziness);
}

__host__ __device__ material* scene_factory::createDielectricMaterial(
    const char* name,
    float refraction,
    bool debug)
{
    if (debug)
        printf("[GPU] dielectricMaterial %s %g\n",
            name,
            refraction);

    return new dielectric(refraction);
}

__host__ __device__ material* scene_factory::createIsotropicMaterial(
    const char* name,
    const color& rgb,
    bool debug)
{
    if (debug)
        printf("[GPU] isotropicMaterial %s %g/%g/%g\n",
            name,
            rgb.r(), rgb.g(), rgb.b());

    return new isotropic(rgb);
}

__host__ __device__ material* scene_factory::createIsotropicMaterial(
    const char* name,
    const char* textureName,
    texture* texture,
    bool debug)
{
    if (debug)
        printf("[GPU] isotropicMaterial %s %s\n",
            name,
            textureName);

    return new isotropic(texture);
}

__host__ __device__ material* scene_factory::createAnisotropicMaterial(
    const char* name,
    float nuf,
    float nvf,
    const char* diffuseTextureName,
    texture* diffuseTexture,
    const char* specularTextureName,
    texture* specularTexture,
    const char* exponentTextureName,
    texture* exponentTexture,
    bool debug)
{
    if (debug)
        printf("[GPU] anisotropicMaterial %s %g %g %s %s %s\n",
            name,
            nuf,
            nvf,
            diffuseTextureName,
            specularTextureName,
            exponentTextureName);
    
    return new anisotropic(nuf, nvf, diffuseTexture, specularTexture, exponentTexture);
}

__host__ __device__ material* scene_factory::createOrenNayarMaterial(
    const char* name,
    const color& rgb,
    float roughness,
    float albedo_temp,
    bool debug)
{
    if (debug)
        printf("[GPU] orenNayarMaterial %s %g/%g/%g %g %g\n",
            name,
            rgb.r(), rgb.g(), rgb.b(),
            roughness,
            albedo_temp);

    return new oren_nayar(rgb, roughness, albedo_temp);
}

__host__ __device__ material* scene_factory::createOrenNayarMaterial(
    const char* name,
    const char* textureName,
    texture* texture,
    float roughness,
    float albedo_temp,
    bool debug)
{
    if (debug)
        printf("[GPU] orenNayarMaterial %s %s %g %g\n",
            name,
            textureName,
            roughness,
            albedo_temp);

    return new oren_nayar(texture, roughness, albedo_temp);
}

__host__ __device__ material* scene_factory::createPhongMaterial(
    const char* name,
    const char* diffuseTextureName,
    texture* diffuseTexture,
    const char* specularTextureName,
    texture* specularTexture,
    const char* bumpTextureName,
    texture* bumpTexture,
    const char* normalTextureName,
    texture* normalTexture,
    const char* displaceTextureName,
    texture* displaceTexture,
    const char* alphaTextureName,
    texture* alphaTexture,
    const char* emissiveTextureName,
    texture* emissiveTexture,
    const color& ambientColor,
    float shininess,
    bool debug)
{
    if (debug)
        printf("[GPU] phongMaterial %s %s %s %s %s %s %s %s %g/%g/%g %g\n",
            name,
            diffuseTextureName,
            specularTextureName,
            bumpTextureName,
            normalTextureName,
            displaceTextureName,
            alphaTextureName,
            emissiveTextureName,
            ambientColor.r(), ambientColor.g(), ambientColor.b(),
            shininess);

    return new phong(diffuseTexture, specularTexture, bumpTexture, normalTexture, displaceTexture, alphaTexture, emissiveTexture, ambientColor, shininess);
}











__host__ __device__ texture* scene_factory::createColorTexture(
    const char* name,
    const color& rgb,
    bool debug)
{
    if (debug)
        printf("[GPU] solidColorTexture %s %g/%g/%g\n",
            name,
            rgb.r(), rgb.g(), rgb.b());
    
    return new solid_color_texture(rgb);
}

__host__ __device__ texture* scene_factory::createGradientTexture(
    const char* name,
    const color& color1,
    const color& color2,
    bool vertical,
    bool hsv,
    bool debug)
{
    if (debug)
        printf("[GPU] gradientColorTexture %s %g/%g/%g %g/%g/%g %d %d\n",
            name,
            color1.r(), color1.g(), color1.b(),
            color2.r(), color2.g(), color2.b(),
            vertical,
            hsv);

    return new gradient_texture(color1, color2, vertical, hsv);
}

__host__ __device__ texture* scene_factory::createImageTexture(const char* name, const char* filepath, const bitmap_image& img, bool debug)
{
    if (debug)
        printf("[GPU] imageTexture %s %s\n",
            name,
            filepath);
    
    return new image_texture(img);
}

__host__ __device__ texture* scene_factory::createCheckerTexture(const char* name, const color& oddColor, const color& evenColor, texture* oddTexture, texture* evenTexture, float scale, bool debug)
{
    if (debug)
        printf("[GPU] checkerTexture %s %g/%g/%g %g/%g/%g %g\n",
            name,
            oddColor.r(), oddColor.g(), oddColor.b(),
            evenColor.r(), evenColor.g(), evenColor.b(),
            scale);
    
    return new checker_texture(scale, oddColor, evenColor);
}

__host__ __device__ texture* scene_factory::createCheckerTexture(const char* name, texture* oddTexture, const char* oddTextureName, texture* evenTexture, const char* eventTextureName, float scale, bool debug)
{
    if (debug)
        printf("[GPU] checkerTexture %s %s %s %g\n",
            name,
            oddTextureName,
            eventTextureName,
            scale);

    return new checker_texture(scale, oddTexture, evenTexture);
}

__host__ __device__ texture* scene_factory::createNoiseTexture(const char* name, float scale, bool debug)
{
    if (debug)
        printf("[GPU] noiseTexture %s %g\n",
            name,
            scale);
    
    return new perlin_noise_texture(scale);
}

__host__ __device__ texture* scene_factory::createBumpTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug)
{
    if (debug)
        printf("[GPU] bumpTexture %s %s %g\n",
            name,
            filepath,
            strength);
    
    return new bump_texture(new image_texture(img), strength);
}


__host__ __device__ texture* scene_factory::createNormalTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug)
{
    if (debug)
        printf("[GPU] normalTexture %s %s %g\n",
            name,
            filepath,
            strength);

    return new normal_texture(new image_texture(img), strength);
}

__host__ __device__ texture* scene_factory::createDisplaceTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug)
{
    if (debug)
        printf("[GPU] displaceTexture %s %s %g\n",
            name,
            filepath,
            strength);

    return new displacement_texture(new image_texture(img), strength);
}

__host__ __device__ texture* scene_factory::createAlphaTexture(const char* name, const char* filepath, const bitmap_image& img, bool doudle_sided, bool debug)
{
    if (debug)
        printf("[GPU] alphaTexture %s %s %d\n",
            name,
            filepath,
            doudle_sided);

    return new alpha_texture(new image_texture(img), doudle_sided);
}

__host__ __device__ texture* scene_factory::createEmissiveTexture(const char* name, const char* filepath, const bitmap_image& img, float strength, bool debug)
{
    if (debug)
        printf("[GPU] emisiveTexture %s %s %g\n",
            name,
            filepath,
            strength);

    return new emissive_texture(new image_texture(img), strength);
}


__host__ __device__ hittable* scene_factory::applyTransform(hittable* primitive, const rt::transform& trs)
{
    if (primitive)
    {
        if (trs.hasTranslate())
            primitive = new rt::translate(primitive, trs.getTranslate());

        if (trs.hasRotate())
            primitive = new rt::rotate(primitive, trs.getRotate());

        if (trs.hasScale())
            primitive = new rt::scale(primitive, trs.getScale());
    }

    return primitive;
}

__host__ __device__ hittable* scene_factory::createDirLight(const char* name, const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible, bool debug)
{
    if (debug)
        printf("[GPU] dirlight %g %s %g/%g/%g %g/%g/%g %g/%g/%g %g/%g/%g %d\n",
            intensity, name,
            pos.x, pos.y, pos.z,
            u.x, u.y, u.z,
            v.x, v.y, v.z,
            rgb.r(), rgb.g(), rgb.b(),
            invisible);
    
    return new directional_light(pos, u, v, intensity, rgb, name, invisible);
}

__host__ __device__ hittable* scene_factory::createOmniLight(const char* name, const point3& pos, float radius, float intensity, color rgb, bool invisible, bool debug)
{
    if (debug)
        printf("[GPU] omnilight %g %s %g/%g/%g %g/%g/%g %g %d\n",
            intensity, name,
            pos.x, pos.y, pos.z,
            rgb.r(), rgb.g(), rgb.b(),
            radius,
            invisible);
    
    return new omni_light(pos, radius, intensity, rgb, name, invisible);
}

__host__ __device__ hittable* scene_factory::createSpotLight(const char* name, const point3& pos, const vector3& dir, float cutoff, float falloff, float intensity, float radius, color rgb, bool invisible, bool debug)
{
    if (debug)
        printf("[GPU] spotlight %g %s %g/%g/%g %g/%g/%g %g %g %g %g/%g/%g %d\n",
            intensity, name,
            pos.x, pos.y, pos.z,
            dir.x, dir.y, dir.z,
            cutoff,
            falloff,
            radius,
            rgb.r(), rgb.g(), rgb.b(),
            invisible);
    
    return new spot_light(pos, dir, cutoff, falloff, intensity, radius, rgb, name, invisible);
}

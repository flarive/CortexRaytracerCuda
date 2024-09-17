#include "scene_factory.h"


#include "../utilities/mesh_loader.cuh"




#include "../primitives/box.cuh"
#include "../primitives/cone.cuh"
#include "../primitives/sphere.cuh"
#include "../primitives/cylinder.cuh"
#include "../primitives/disk.cuh"
#include "../primitives/torus.cuh"
#include "../primitives/aarect.cuh"
#include "../primitives/quad.cuh"
#include "../primitives/volume.cuh"

#include "../lights/directional_light.cuh"
#include "../lights/omni_light.cuh"
#include "../lights/spot_light.cuh"




hittable* scene_factory::createBox(
        const char* name,
        const point3 &p0,
        const point3 &p1,
        material* material,
        const uvmapping& uv)
{
    return new box(p0, p1, material, uv, name);
}

hittable* scene_factory::createCylinder(
        const char* name,
        const point3 &center,
        float radius,
        float height,
        material* material,
        const uvmapping& uv)
{
    return new cylinder(center, radius, height, material, uv);
}

hittable* scene_factory::createDisk(
        const char* name,
        const point3& center,
        float radius,
        float height,
        material* material,
        const uvmapping& uv)
{
    return new disk(center, radius, height, material, uv);
}

hittable* scene_factory::createTorus(
        const char* name,
        const point3& center,
        float major_radius,
        float minor_radius,
        material* material,
        const uvmapping& uv)
{
    return new torus(center, major_radius, minor_radius, material, uv);
}

hittable* scene_factory::createSphere(
        const char* name,
        const point3& center,
        float radius,
        material* material,
        const uvmapping& uv)
{
    return new sphere(center, radius, material, uv, name);
}

hittable* scene_factory::createCone(
        const char* name,
        const point3& center,
        float height,
        float radius,
        material* material,
        const uvmapping& uv)
{
    return new cone(center, radius, height, material, uv, name);
}

hittable* scene_factory::createPlane(
    const char* name,
    const point3 &p0,
    point3 p1,
    material* material,
    const uvmapping& uv)
{
    if (p0.x == p1.x)
    {
        float x = p0.x;
        float y0 = p0.y < p1.y ? p0.y : p1.y;
        float y1 = p0.y < p1.y ? p1.y : p0.y;
        float z0 = p0.z < p1.z ? p0.z : p1.z;
        float z1 = p0.z < p1.z ? p1.z : p0.z;

        return new yz_rect(y0, y1, z0, z1, x, material, uv);
    }

    if (p0.y == p1.y)
    {
        float y = p0.y;
        float x0 = p0.x < p1.x ? p0.x : p1.x;
        float x1 = p0.x < p1.x ? p1.x : p0.x;
        float z0 = p0.z < p1.z ? p0.z : p1.z;
        float z1 = p0.z < p1.z ? p1.z : p0.z;

        return new xz_rect(x0, x1, z0, z1, y, material, uv);
    }

    if (p0.z == p1.z)
    {
        float z = p0.z;
        float x0 = p0.x < p1.x ? p0.x : p1.x;
        float x1 = p0.x < p1.x ? p1.x : p0.x;
        float y0 = p0.y < p1.y ? p0.y : p1.y;
        float y1 = p0.y < p1.y ? p1.y : p0.y;

        return new xy_rect(x0, x1, y0, y1, z, material, uv);
    }

    throw std::runtime_error("a plane should always be created aligned to one of the x, y, or z axes");
}

hittable* scene_factory::createQuad(
    const char* name,
    const point3& position,
    const vector3 u,
    const vector3 v,
    material* material,
    const uvmapping& uv)
{
    return new quad(position, u, v, material, uv, name);
}

hittable* scene_factory::createVolume(
    const char* name,
    hittable* boundary,
    float density,
    texture* texture)
{
    return new volume(boundary, density, texture, name);
}

hittable* scene_factory::createVolume(
    const char* name,
    hittable* boundary,
    float density,
    const color& rgb)
{
    return new volume(boundary, density, rgb, name);
}

hittable* scene_factory::createMesh(
    const char* name,
	const point3& center,
	const char* filepath,
	material* material,
	const bool use_mtl,
    const bool use_smoothing)
{
    hittable* mesh = nullptr;
    
    mesh_loader::mesh_data data;
    
    if (mesh_loader::load_model_from_file(filepath, data))
    {
        mesh = mesh_loader::convert_model_from_file(data, material, use_mtl, use_smoothing, name);
    }

    return mesh;
}

hittable* scene_factory::createDirectionalLight(const char* name, const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible)
{
    return new directional_light(pos, u, v, intensity, rgb, name, invisible);
}

hittable* scene_factory::createOmniDirectionalLight(const char* name, const point3& pos, float radius, float intensity, color rgb, bool invisible)
{
    return new omni_light(pos, radius, intensity, rgb, name, invisible);
}

hittable* scene_factory::createSpotLight(const char* name, const point3& pos, const vector3& dir, float cutoff, float falloff, float intensity, float radius, color rgb, bool invisible)
{
    return new spot_light(pos, dir, cutoff, falloff, intensity, radius, rgb, name, invisible);
}


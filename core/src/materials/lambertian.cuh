#pragma once

#include "material.cuh"
#include "../misc/gpu_randomizer.cuh"
#include "../primitives/hittable.cuh"
#include "../textures/texture.cuh"

class lambertian : public material
{
public:
    __device__ lambertian(texture* a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const
    {
        vector3 target = rec.hit_point + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.hit_point, target - rec.hit_point, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.hit_point);
        return true;
    }

    texture* albedo;
};
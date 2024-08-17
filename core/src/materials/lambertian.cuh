#pragma once

#include "material.cuh"
#include "../misc/gpu_randomizer.cuh"
#include "../primitives/hittable.cuh"
#include "../textures/texture.cuh"

class lambertian : public material
{
public:
    __device__ lambertian(texture* a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const {
        vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    texture* albedo;
};
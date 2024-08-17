#pragma once

#include "../misc/randomizer.cuh"

class isotropic : public material
{
public:
    __device__ isotropic(texture* a) : albedo(a) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState* local_rand_state) const {
        scattered = ray(rec.p, random_in_unit_sphere(local_rand_state), r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    texture* albedo;
};
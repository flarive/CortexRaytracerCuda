#pragma once

#include "material.cuh"
#include "../textures/texture.cuh"

class diffuse_light : public material
{
public:
    __device__ diffuse_light(texture* tex) : emit(tex) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const {
        return false;
    }
    __device__ vector3 emitted(float u, float v, const vector3& p) const {
        return emit->value(u, v, p);
    }

    texture* emit;
};
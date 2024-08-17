#pragma once

#include "../primitives/hittable.cuh"

class material
{
public:
    __device__ virtual vector3 emitted(float u, float v, const vector3& p) const { return vector3(0,0,0); }
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};
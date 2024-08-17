#pragma once

#include "../misc/vector3.cuh"

//#include "cuda_runtime.h"
//#include <curand_kernel.h>


__device__ inline vector3 random_in_unit_disk(curandState* local_rand_state)
{
    vector3 p;
    do {
        p = 2.0f * vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vector3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}


__device__ inline vector3 random_in_unit_sphere(curandState* local_rand_state)
{
    vector3 p;
    do {
        p = 2.0f * vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vector3(1, 1, 1);
    } while (l1(p) >= 1.0f);
    return p;
}

__device__ inline bool refract(const vector3& v, const vector3& n, float ni_over_nt, vector3& refracted) {
    vector3 uv = unitv(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}



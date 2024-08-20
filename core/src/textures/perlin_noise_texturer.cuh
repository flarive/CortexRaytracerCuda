#pragma once

#include "texture.cuh"
#include "perlin.cuh"
#include "../misc/vector3.cuh"


class perlin_noise : public texture
{
public:
    __device__ perlin_noise(int scale, curandState* local_rand_state) : scale(scale), noise(perlin(local_rand_state)) { }
    __device__ virtual vector3 value(float u, float v, const point3& p) const {
        return vector3(1, 1, 1) * noise.noise(scale, p);
    }

    perlin noise;
    int scale;
};
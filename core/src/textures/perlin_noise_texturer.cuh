#pragma once

#include "texture.cuh"
#include "perlin.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"


class perlin_noise : public texture
{
public:
    __device__ perlin_noise(int scale, curandState* local_rand_state) : scale(scale), noise(perlin(local_rand_state)) { }

    __device__ virtual color value(float u, float v, const point3& p) const {
        return color(1, 1, 1) * noise.noise(scale, p);
    }

    perlin noise;
    int scale;
};
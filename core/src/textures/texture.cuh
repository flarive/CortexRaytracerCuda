#pragma once

#include "perlin.cuh"

class texture
{
public:
    __host__ __device__ virtual color value(float u, float v, const point3& p) const = 0;
};
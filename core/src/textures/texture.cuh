#pragma once

#include "perlin.cuh"

class texture
{
public:
    __device__ virtual color value(float u, float v, const vector3& p) const = 0;
};
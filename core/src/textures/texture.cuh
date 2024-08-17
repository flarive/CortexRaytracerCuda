#pragma once

#include "perlin.cuh"

class texture
{
public:
    __device__ virtual vector3 value(float u, float v, const vector3& p) const = 0;
};
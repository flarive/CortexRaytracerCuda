#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"

class checker_texture : public texture
{
public:
    __device__ checker_texture() {}
    __device__ checker_texture(texture* t0, texture* t1) : even(t0), odd(t1) {}
    __device__ virtual vector3 value(float u, float v, const vector3& p) const {
        float sines = sin(10 * p.x) * sin(10 * p.y) * sin(10 * p.z);
        if (sines < 0) {
            return odd->value(u, v, p);
        }
        else {
            return even->value(u, v, p);
        }
    }

    texture* odd;
    texture* even;
};
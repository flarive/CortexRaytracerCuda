#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"


class solid_color_texture : public texture
{
public:
    __device__ solid_color_texture() {}
    __device__ solid_color_texture(vector3 c) : color(c) {}
    __device__ virtual vector3 value(float u, float v, const point3& p) const {
        return color;
    }

    vector3 color;
};

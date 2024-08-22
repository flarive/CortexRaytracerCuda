#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"


class solid_color_texture : public texture
{
public:
    __device__ solid_color_texture() {}
    __device__ solid_color_texture(color c) : m_color(c) {}
    __device__ virtual color value(float u, float v, const point3& p) const {
        return m_color;
    }

    color m_color;
};

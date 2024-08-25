#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"

/// <summary>
/// Solid color texture
/// </summary>
class solid_color_texture : public texture
{
public:
    //__host__ __device__ solid_color_texture() {}
    //__host__ __device__ solid_color_texture(color c) : m_color(c) {}
    //__device__ virtual color value(float u, float v, const point3& p) const {
    //    return m_color;
    //}

    __host__ __device__ solid_color_texture(color c);
    __host__ __device__ solid_color_texture(float red, float green, float blue);

    __host__ __device__ color value(float u, float v, const point3& p) const override;

    __host__ __device__ color get_color() const;

private:
    color m_color_value{};
};


__host__ __device__ solid_color_texture::solid_color_texture(color c) : m_color_value(c)
{
}

__host__ __device__ solid_color_texture::solid_color_texture(float red, float green, float blue) : solid_color_texture(color(red, green, blue))
{
}

__host__ __device__ color solid_color_texture::value(float u, float v, const point3& p) const
{
    return m_color_value;
}

__host__ __device__ color solid_color_texture::get_color() const
{
    return m_color_value;
}

#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"

class checker_texture : public texture
{
public:
    __host__ __device__ checker_texture(float _scale, texture* _even, texture* _odd);
    __host__ __device__ checker_texture(float _scale, color c1, color c2);

    __host__ __device__ color value(float u, float v, const point3& p) const override;

private:
    float m_scale = 0.0f;
    float m_inv_scale = 0.0f;
    texture* m_even;
    texture* m_odd;
};

__host__ __device__ checker_texture::checker_texture(float _scale, texture* _even, texture* _odd)
    : m_scale(_scale), m_inv_scale(1.0f / _scale), m_even(_even), m_odd(_odd)
{
}

__host__ __device__ checker_texture::checker_texture(float _scale, color c1, color c2)
    : m_scale(_scale), m_inv_scale(1.0f / _scale), m_even(new solid_color_texture(c1)), m_odd(new solid_color_texture(c2))
{
}

__host__ __device__ color checker_texture::value(float u, float v, const point3& p) const
{
    auto xInteger = static_cast<int>(std::floor(m_inv_scale * p.x));
    auto yInteger = static_cast<int>(std::floor(m_inv_scale * p.y));
    auto zInteger = static_cast<int>(std::floor(m_inv_scale * p.z));

    bool isEven = (xInteger + yInteger + zInteger) % 2 == 0;

    return isEven ? m_even->value(u, v, p) : m_odd->value(u, v, p);
}
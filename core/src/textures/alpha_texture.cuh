#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Alpha texture
/// </summary>
class alpha_texture : public texture
{
public:
    __host__ __device__ alpha_texture(texture* alpha, bool double_sided);

    __host__ __device__ color value(float u, float v, const point3& p) const;

    __host__ __device__ bool is_double_sided() const;

    __host__ __device__ TextureTypeID getTypeID() const override { return TextureTypeID::textureAlphaType; }


private:
    texture* m_alpha;
    bool m_double_sided = false;
};


__host__ __device__ inline alpha_texture::alpha_texture(texture* alpha, bool double_sided = false) : m_alpha(alpha), m_double_sided(double_sided)
{
}

__host__ __device__ inline color alpha_texture::value(float u, float v, const point3& p) const
{
    return m_alpha->value(u, v, p);
}

__host__ __device__ inline bool alpha_texture::is_double_sided() const
{
    return m_double_sided;
}
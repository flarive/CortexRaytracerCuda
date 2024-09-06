#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Normal texture
/// </summary>
class normal_texture : public texture
{
public:
    __host__ __device__ normal_texture(texture* normal, float strength = 10.0f);

    __host__ __device__ color value(float u, float v, const point3& p) const override;

    __host__ __device__ float getStrenth();

    __host__ __device__ TextureTypeID getTypeID() const override { return TextureTypeID::textureNormalType; }

private:
    texture* m_normal;
    float m_strength = 10.0f;
};



__host__ __device__ normal_texture::normal_texture(texture* normal, float strength) : m_normal(normal), m_strength(strength)
{
}

__host__ __device__ color normal_texture::value(float u, float v, const point3& p) const
{
    // Sample the underlying texture to get the normal map
    color normal_map = m_normal->value(u, v, p);

    // Scale from [0, 1] to [-1, 1]
    vector3 normal;
    normal.x = 2.0f * normal_map.r() - 1.0f;
    normal.y = 2.0f * normal_map.g() - 1.0f;
    normal.z = 2.0f * normal_map.b() - 1.0f;

    // Normalize the resulting vector
    normal = glm::normalize(normal);

    // Convert normalized vector back to color for output
    return color(normal.x, normal.y, normal.z);
}

__host__ __device__ float normal_texture::getStrenth()
{
    return m_strength;
}

#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Amissive texture
/// </summary>
class emissive_texture : public texture
{
public:
	__host__ __device__ emissive_texture();
	__host__ __device__ emissive_texture(texture* bump, float strength = 10.0f);

	__host__ __device__ color value(float u, float v, const point3& p) const;

	__host__ __device__ TextureTypeID getTypeID() const override { return TextureTypeID::textureEmissiveType; }

private:
	texture* m_emissive;
	float m_strength = 10.0f;     // Scaling factor for emissive effect
};

__host__ __device__ emissive_texture::emissive_texture()
{
}

__host__ __device__ emissive_texture::emissive_texture(texture* bump, float strength) : m_emissive(bump), m_strength(strength)
{
}

__host__ __device__ color emissive_texture::value(float u, float v, const point3& p) const
{
	return m_emissive->value(u, v, p);
}
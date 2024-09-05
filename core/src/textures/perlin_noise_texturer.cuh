#pragma once

#include "texture.cuh"
#include "perlin.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Perlin noise texture
/// </summary>
class perlin_noise_texture : public texture
{
public:
    __host__ __device__ perlin_noise_texture();
    __host__ __device__ perlin_noise_texture(double sc);

    __host__ __device__ color value(float u, float v, const point3& p) const override;

private:
    perlin noise;
    float scale = 0.0f;
};


__host__ __device__ perlin_noise_texture::perlin_noise_texture()
{

}

__host__ __device__ perlin_noise_texture::perlin_noise_texture(double sc) : scale(sc)
{

}

__host__ __device__ color perlin_noise_texture::value(float u, float v, const point3& p) const
{
    auto s = scale * p;
    return color(1, 1, 1) * 0.5f * (1 + sin(s.z + 10 * noise.turb(s)));
}
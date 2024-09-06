#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Gradient texture
/// </summary>
class gradient_texture : public texture
{
public:
    __host__ __device__ gradient_texture();
    __host__ __device__ gradient_texture(color c1, color c2, bool v, bool hsv2);

    __host__ __device__ virtual color value(float u, float v, const point3& p) const;

    __host__ __device__ TextureTypeID getTypeID() const override { return TextureTypeID::textureGradientType; }

private:
    color gamma_color1{}, gamma_color2{};
    bool aligned_v = false;
    bool hsv = false;
};


__host__ __device__ gradient_texture::gradient_texture()
{
}

__host__ __device__ gradient_texture::gradient_texture(color c1, color c2, bool v, bool hsv2) : aligned_v(v)
{
    gamma_color1 = hsv2 ? color::RGBtoHSV(c1) : c1;
    gamma_color2 = hsv2 ? color::RGBtoHSV(c2) : c2;
    hsv = hsv2;
}

__host__ __device__ color gradient_texture::value(float u, float v, const point3& p) const
{
    color final_color = aligned_v ? gamma_color1 * (1 - u) + u * gamma_color2 : gamma_color1 * (1 - v) + v * gamma_color2;

    return (hsv ? color::HSVtoRGB(final_color) : final_color);
}

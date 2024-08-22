#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"

class image_texture : public texture
{
public:
    __device__ image_texture() {}
    __device__ image_texture(unsigned char* pixels, int A, int B) : data(pixels), nx(A), ny(B) {}
    __device__ virtual color value(float u, float v, const point3& p) const;

    unsigned char* data;
    int nx, ny;
};

__device__ color image_texture::value(float u, float v, const point3& p) const
{
    int i = u * nx;
    int j = (1 - v) * ny - 0.001;
    if (i < 0) i = 0;
    if (j < 0) j = 0;
    if (i > nx - 1) i = nx - 1;
    if (j > ny - 1) j = ny - 1;
    float r = int(data[3 * i + 3 * nx * j]) / 255.0;
    float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
    float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
    return color(r, g, b);
}
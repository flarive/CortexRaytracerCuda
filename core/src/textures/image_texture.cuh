#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../utilities/bitmap_image.cuh"

class image_texture : public texture
{
public:
    __host__ __device__ image_texture() {}
    __host__ __device__ image_texture(bitmap_image img) : m_image(img) {}


    __host__ __device__ color value(float u, float v, const point3& p) const override;


    __host__ __device__ inline int getWidth() const
    {
        return m_image.width();
    }

    __host__ __device__ inline int getHeight() const
    {
        return m_image.height();
    }

    __host__ __device__ inline int getChannels() const
    {
        return m_image.channels();
    }

    __host__ __device__ inline unsigned char* get_data() const
    {
        return m_image.get_data();
    }

    __host__ __device__ inline float* get_data_float() const
    {
        return m_image.get_data_float();
    }

private:
    bitmap_image m_image;
};





__host__ __device__ inline color image_texture::value(float u, float v, const point3& p) const
{
    // If we have no texture data, then return solid cyan as a debugging aid.
    if (m_image.height() <= 0) return color(0, 1, 1);

    // Clamp input texture coordinates to [0,1] x [1,0]
    u = interval(0, 1).clamp(u);
    v = 1.0f - interval(0, 1).clamp(v);  // Flip V to image coordinates

    auto i = static_cast<int>(u * m_image.width());
    auto j = static_cast<int>(v * m_image.height());

    float color_scale = 1.0f / 255.0f;
    const unsigned char* pixel = m_image.pixel_data(i, j);
    return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}


//__device__ color image_texture::value(float u, float v, const point3& p) const
//{
//    int i = u * nx;
//    int j = (1 - v) * ny - 0.001;
//    if (i < 0) i = 0;
//    if (j < 0) j = 0;
//    if (i > nx - 1) i = nx - 1;
//    if (j > ny - 1) j = ny - 1;
//    float r = int(data[3 * i + 3 * nx * j]) / 255.0;
//    float g = int(data[3 * i + 3 * nx * j + 1]) / 255.0;
//    float b = int(data[3 * i + 3 * nx * j + 2]) / 255.0;
//    return color(r, g, b);
//}
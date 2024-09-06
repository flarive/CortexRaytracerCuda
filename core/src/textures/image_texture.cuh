#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../utilities/bitmap_image.cuh"

class image_texture : public texture
{
public:
    //__host__ __device__ image_texture() {}
    __host__ __device__ image_texture(bitmap_image img) : m_image(img) {}

    __host__ __device__ virtual TextureTypeID getTypeID() const { return TextureTypeID::textureImageType; }

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


    int i = static_cast<int>(u * m_image.width());
    int j = static_cast<int>(v * m_image.height());

    float color_scale = 1.0f / 255.0f;
    const unsigned char* pixel = m_image.pixel_data(i, j);

    return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
}
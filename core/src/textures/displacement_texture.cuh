#pragma once

#include "texture.cuh"
#include "image_texture.cuh"
#include "../misc/vector3.cuh"

/// <summary>
/// Displacement texture
/// https://stackoverflow.com/questions/4476669/ray-tracing-a-sphere-with-displacement-mapping
/// </summary>
class displacement_texture : public texture
{
public:
    __host__ __device__ displacement_texture();

    __host__ __device__ displacement_texture(texture* t, float strength);

    __host__ __device__ color value(float u, float v, const point3& p) const override;

    __host__ __device__ float getStrenth();

    __host__ __device__ TextureTypeID getTypeID() const override { return TextureTypeID::textureDisplacementType; }

private:
    texture* m_displacement = nullptr;
    float m_strength = 10.0f;

    int m_width = 0;
    int m_height = 0;
    int m_channels = 3;

    float* m_data = nullptr;
};


__host__ __device__ inline displacement_texture::displacement_texture()
{
}

__host__ __device__ inline displacement_texture::displacement_texture(texture* t, float strength)
    : m_displacement(t), m_strength(strength)
{
    //image_texture* imageTex = dynamic_cast<image_texture*>(m_displacement);
    //if (imageTex)
    //{
    //    m_width = imageTex->getWidth();
    //    m_height = imageTex->getHeight();
    //    m_data = imageTex->get_data_float();
    //    m_channels = imageTex->getChannels();
    //}

    if (m_displacement->getTypeID() == TextureTypeID::textureDisplacementType)
    {
        image_texture* imageTex = static_cast<image_texture*>(m_displacement);
        if (imageTex)
        {
            m_width = imageTex->getWidth();
            m_height = imageTex->getHeight();
            m_data = imageTex->get_data_float();
            m_channels = imageTex->getChannels();
        }
    }
}

__host__ __device__ inline color displacement_texture::value(float u, float v, const point3& p) const
{
    double value = 0.0;

    // Clamp u and v to [0, 1]
    u = std::fmod(u, 1.0f);
    v = std::fmod(v, 1.0f);

    // Map u and v to texture coordinates
    int x = static_cast<int>(u * m_width) % m_width;
    int y = static_cast<int>(v * m_height) % m_height;

    // Compute the index in the data array
    int index = (y * m_width + x) * m_channels;

    // Calculate the displacement value based on the number of channels
    if (m_channels == 1) { // Grayscale
        value = m_data[index] / m_strength;
    }
    else if (m_channels == 3) { // RGB
        float r = m_data[index] / m_strength;
        float g = m_data[index + 1] / m_strength;
        float b = m_data[index + 2] / m_strength;
        // Using the average of the RGB values as the displacement
        value = (r + g + b) / 3.0f;
    }
    else if (m_channels == 4) { // RGBA
        float r = m_data[index] / m_strength;
        float g = m_data[index + 1] / m_strength;
        float b = m_data[index + 2] / m_strength;
        // Ignore the alpha channel for displacement
        value = (r + g + b) / 3.0f;
    }

    // return a single double as a color
    return color(value, value, value);
}

__host__ __device__ inline float displacement_texture::getStrenth()
{
    return m_strength;
}

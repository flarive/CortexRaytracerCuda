#pragma once

#include "texture.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

/// <summary>
/// Bump texture
/// </summary>
class bump_texture : public texture
{
public:
	//bump_texture();
	__host__ __device__ bump_texture(texture* bump, float strength = 0.5f);

	__host__ __device__ color value(float u, float v, const point3& p) const;

	__host__ __device__ vector3 perturb_normal(const vector3& normal, float u, float v, const vector3& p) const;
	__host__ __device__ vector3 convert_to_world_space(const vector3& tangentSpaceNormal, const vector3& originalNormal) const;
	__host__ __device__ void compute_tangent_space(const vector3& normal, vector3& tangent, vector3& bitangent) const;

    __host__ __device__ virtual TextureTypeID getTypeID() const { return TextureTypeID::textureBaseType; }

private:
	texture* m_bump;
	float m_strength = 0.5f; // normalized and can be between 0.0 and 1.0 (0.5 is usually good)
};



//__host__ __device__ bump_texture::bump_texture()
//{
//}

__host__ __device__ inline bump_texture::bump_texture(texture* bump, float strength) : m_bump(bump), m_strength(strength * 10.0f)
{
}

__host__ __device__ inline color bump_texture::value(float u, float v, const point3& p) const
{
    return m_bump->value(u, v, p);
}

__host__ __device__ inline vector3 bump_texture::perturb_normal(const vector3& normal, float u, float v, const vector3& p) const
{
    float m_bump_width = 0.0f; // doesn't change anything ?
    float m_bump_height = 0.0f; // doesn't change anything ?

    //std::shared_ptr<image_texture> imageTex = std::dynamic_pointer_cast<image_texture>(m_bump);
    //if (imageTex)
    //{
    //    m_bump_width = imageTex->getWidth();
    //    m_bump_height = imageTex->getHeight();
    //}

    if (m_bump->getTypeID() == TextureTypeID::textureBumpType)
    {
        image_texture* imageTex = static_cast<image_texture*>(m_bump);
        if (imageTex)
        {
            m_bump_width = (float)imageTex->getWidth();
            m_bump_height = (float)imageTex->getHeight();
        }
    }


    float heightL = m_bump->value(u - 1.0f / m_bump_width, v, p).r();
    float heightR = m_bump->value(u + 1.0f / m_bump_width, v, p).r();
    float heightD = m_bump->value(u, v - 1.0f / m_bump_height, p).r();
    float heightU = m_bump->value(u, v + 1.0f / m_bump_height, p).r();

    // Scale the height differences using m_scale
    float scaledHeightL = m_strength * heightL;
    float scaledHeightR = m_strength * heightR;
    float scaledHeightD = m_strength * heightD;
    float scaledHeightU = m_strength * heightU;

    vector3 tangentSpaceNormal = glm::normalize(vector3(
        scaledHeightR - scaledHeightL,
        scaledHeightU - scaledHeightD,
        1.0f
    ));

    return convert_to_world_space(tangentSpaceNormal, normal);
}

__host__ __device__ inline vector3 bump_texture::convert_to_world_space(const vector3& tangentSpaceNormal, const vector3& originalNormal) const
{
    vector3 tangent, bitangent;
    compute_tangent_space(originalNormal, tangent, bitangent);

    vector3 worldNormal = glm::normalize(
        tangent * tangentSpaceNormal.x +
        bitangent * tangentSpaceNormal.y +
        originalNormal * tangentSpaceNormal.z
    );

    return worldNormal;
}

__host__ __device__ inline void bump_texture::compute_tangent_space(const vector3& normal, vector3& tangent, vector3& bitangent) const
{
    if (fabs(normal.x) > fabs(normal.y))
    {
        tangent = glm::normalize(glm::vec3(normal.z, 0.0f, -normal.x));
    }
    else
    {
        tangent = glm::normalize(glm::vec3(0.0f, -normal.z, normal.y));
    }
    bitangent = glm::normalize(glm::cross(normal, tangent));
}
#pragma once

#include "material.cuh"
#include "../textures/texture.cuh"

class diffuse_spot_light : public material
{
public:
    __host__ __device__ diffuse_spot_light(texture* emitTex, point3 pos, vector3 dir, float cutoff, float falloff, float intensity, bool invisible);

    __host__ __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, curandState* local_rand_state) const override;

private:
    texture* m_emit = nullptr;
    point3 m_position{};
    float m_intensity = 1.0f;
    bool m_invisible = true;
    bool m_directional = true;
    vector3 m_direction = vector3(0, 0, -1); // Default direction
    float m_cutoff = cos(degrees_to_radians(30.0f)); // Cutoff angle in radians
    float m_falloff = 1.0f; // Falloff exponent
};


__host__ __device__ diffuse_spot_light::diffuse_spot_light(texture* emitTex, point3 pos, vector3 dir, float cutoff, float falloff, float intensity, bool invisible) :
    m_emit(emitTex), m_position(pos), m_direction(dir), m_intensity(intensity), m_cutoff(cutoff),
    m_falloff(falloff), m_invisible(invisible)
{
}

__host__ __device__ color diffuse_spot_light::emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, curandState* local_rand_state) const
{
    if (m_directional && !rec.front_face)
    {
        return m_invisible ? color(0, 0, 0, 0) : color(1, 1, 1, 0);
    }

    vector3 light_dir = unit_vector(m_direction);
    vector3 hit_to_light = unit_vector(r_in.direction());

    float cos_theta = glm::dot(hit_to_light, light_dir);
    if (cos_theta < m_cutoff)
    {
        return color(1, 1, 1, 1);
    }

    float attenuation = glm::pow(cos_theta, m_falloff);
    return m_emit->value(u, v, p) * (m_intensity * attenuation);
}
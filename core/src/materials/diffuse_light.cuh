#pragma once

#include "material.cuh"
#include "../textures/texture.cuh"
#include "../textures/solid_color_texture.cuh"

class diffuse_light : public material
{
//public:
//    __device__ diffuse_light(texture* tex) : emit(tex) {}
//    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const {
//        return false;
//    }
//    __device__ vector3 emitted(float u, float v, const vector3& p) const {
//        return emit->value(u, v, p);
//    }
//
//    texture* emit;

public:
    __host__ __device__ diffuse_light(texture* a)
        : m_emit(a) {}

    __host__ __device__ diffuse_light(color _c)
        : m_emit(new solid_color_texture(_c)), m_intensity(1.0), m_invisible(true), m_directional(true) {}

    __host__ __device__ diffuse_light(color _c, float _intensity, bool _directional, bool _invisible)
        : m_emit(new solid_color_texture(_c)), m_intensity(_intensity), m_directional(_directional), m_invisible(_invisible)
    {
    }

    __device__ color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, curandState* local_rand_state) const override;

    __host__ __device__ MaterialTypeID getTypeID() const override { return MaterialTypeID::materialDiffuseLightType; }


private:
    texture* m_emit = nullptr;
    float m_intensity = 1.0f;
    bool m_directional = true;
    bool m_invisible = true;
};


__device__ inline color diffuse_light::emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, curandState* local_rand_state) const
{
    // Material emission, directional or not
    if (m_directional)
    {
        if (rec.front_face)
        {
            // light
            return m_emit->value(u, v, p) * m_intensity;
        }
        else
        {
            // no light
            return color(0, 0, 0, 0);
        }
    }
    else
    {
        // light
        return m_emit->value(u, v, p) * m_intensity;
    }

    return m_emit->value(u, v, p) * m_intensity;
}
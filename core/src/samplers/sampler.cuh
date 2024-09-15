#pragma once

#include "../misc/vector3.cuh"

/// <summary>
/// Sampler base class
/// https://cgg.mff.cuni.cz/~pepca/lectures/pdf/pg2-13-sampling.en.pdf
/// </summary>
class sampler
{
public:
    __host__ __device__ sampler(const vector3& pixel_delta_u, const vector3& pixel_delta_v, int samples = 50, int spp = 2);

    __device__ virtual vector3 generate_samples(int s_i, int s_j, thrust::default_random_engine& rng) const;


protected:
    int m_samples = 0;
    int m_spp = 0;
    float m_recip_sqrt_spp = 0.0f;

    vector3 m_pixel_delta_u{};
    vector3 m_pixel_delta_v{};
};

__host__ __device__ inline sampler::sampler(const vector3& pixel_delta_u, const vector3& pixel_delta_v, int samples, int spp)
    : m_pixel_delta_u(pixel_delta_u), m_pixel_delta_v(pixel_delta_v), m_samples(samples), m_spp(spp)
{
    int sqrt_spp = static_cast<int>(glm::sqrt(samples));
    m_recip_sqrt_spp = 1.0f / sqrt_spp;
}

__device__ inline vector3 sampler::generate_samples(int s_i, int s_j, thrust::default_random_engine& rng) const
{
    return vector3{};
}
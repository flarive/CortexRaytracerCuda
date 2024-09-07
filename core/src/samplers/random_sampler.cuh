#pragma once

#include "sampler.cuh"

#include "../misc/gpu_randomizer.cuh"


class random_sampler : public sampler
{
public:
	__host__ __device__ random_sampler(const vector3& pixel_delta_u, const vector3& pixel_delta_v, int samples = 50);

	__device__ virtual vector3 generate_samples(int s_i, int s_j, curandState* local_rand_state) const override;
};




__host__ __device__ inline random_sampler::random_sampler(const vector3& pixel_delta_u, const vector3& pixel_delta_v, int samples)
    : sampler(pixel_delta_u, pixel_delta_v, samples)
{
}

/// <summary>
/// Simple random sampler Anti-Aliasing
/// A stochastic sample pattern is a random distribution of multisamples throughout the pixel.
/// The irregular spacing of samples makes attribute evaluation complicated.
/// The method is cost efficient due to low sample count (compared to regular grid patterns).
/// Edge optimization with this method, although sub-optimal for screen aligned edges.
/// Image quality is excellent for a moderate number of samples.
/// </summary>
/// <param name="s_i"></param>
/// <param name="s_j"></param>
/// <returns></returns>
__device__ inline vector3 random_sampler::generate_samples(int s_i, int s_j, curandState* local_rand_state) const
{
    // Generate random positions within the pixel
    float px = -0.5f + m_recip_sqrt_spp * (s_i + get_real(local_rand_state));
    float py = -0.5f + m_recip_sqrt_spp * (s_j + get_real(local_rand_state));
    return (px * m_pixel_delta_u) + (py * m_pixel_delta_v);
}

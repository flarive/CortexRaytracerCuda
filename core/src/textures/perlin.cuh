#pragma once

#include <thrust/random.h>
#include "../misc/gpu_randomizer.cuh"

__host__ __device__ inline float* perlin_generate(thrust::default_random_engine& rng)
{
    float* p = new float[256];
    for (int i = 0; i < 256; ++i)
        p[i] = get_real(rng);
    return p;
}

__host__ __device__ inline void permute(int *p, int n, thrust::default_random_engine& rng)
{
    for (int i = n-1; i > 0; i--) {
        int target = int(get_real(rng)*(i+1));
        int tmp = p[i];
        p[i] = p[target];
        p[target] = tmp;
    }
    return;
}

__host__ __device__ inline int* perlin_generate_perm(thrust::default_random_engine& rng)
{
    int * p = new int[256];
    for (int i = 0; i < 256; i++)
        p[i] = i;
    permute(p, 256, rng);
    return p;
}

class perlin
{
public:
    __host__ __device__ perlin()
    {
        // thrust random engine and distribution
        int seed = 7896333;
        thrust::minstd_rand rng(seed);

        m_ranfloat = perlin_generate(rng);
        m_perm_x = perlin_generate_perm(rng);
        m_perm_y = perlin_generate_perm(rng);
        m_perm_z = perlin_generate_perm(rng);
    }

    __host__ __device__ float turb(int scale, const vector3& p) const
    {
        int x = uint8_t(p.x * scale) % 256;
        int y = uint8_t(p.y * scale) % 256;
        int z = uint8_t(p.z * scale) % 256;
        return m_ranfloat[m_perm_x[x] ^ m_perm_y[y] ^ m_perm_z[z]];
    }

private:
    //thrust::default_random_engine& m_rng;
    float *m_ranfloat;
    int *m_perm_x;
    int *m_perm_y;
    int *m_perm_z;
};
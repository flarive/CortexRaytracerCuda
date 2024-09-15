#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

class sphere_pdf : public pdf
{
public:
    __device__ sphere_pdf() { }

    __device__ ~sphere_pdf() = default;

    __device__ float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const override;
    __device__ vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) override;

    __host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfSphere; }
};

__device__ inline float sphere_pdf::value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const
{
    return 1.0f / (4.0f * M_PI);
}

__device__ inline vector3 sphere_pdf::generate(scatter_record& rec, thrust::default_random_engine& rng)
{
    return get_unit_vector(rng);
}
#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

class sphere_pdf : public pdf
{
public:
    __device__ sphere_pdf() { }

    __device__ ~sphere_pdf() = default;

    __device__ float value(const vector3& direction, curandState* local_rand_state) const override;
    __device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) override;

    __host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfSphere; }
};

__device__ inline float sphere_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
    return 1.0f / (4.0f * M_PI);
}

__device__ inline vector3 sphere_pdf::generate(scatter_record& rec, curandState* local_rand_state)
{
    return get_unit_vector(local_rand_state);
}
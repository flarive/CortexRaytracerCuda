#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

class sphere_pdf : public pdf
{
public:
    __device__ sphere_pdf() { }

    __device__ sphere_pdf* clone() const override {
        return new sphere_pdf(*this);
    }

    __device__ ~sphere_pdf() = default;

    __device__ float value(const vector3& direction, curandState* local_rand_state) const override;
    __device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) const override;

    __device__ pdfTypeID getTypeID() const override { return pdfTypeID::pdfSphere; }
};

__device__ inline float sphere_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
    return 1.0f / (4.0f * M_PI);
}

__device__ inline vector3 sphere_pdf::generate(scatter_record& rec, curandState* local_rand_state) const
{
    return get_unit_vector(local_rand_state);
}
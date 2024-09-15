#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../utilities/uvmapping.cuh"
#include "../misc/gpu_randomizer.cuh"


class hittable_pdf : public pdf
{
public:
    __device__ hittable_pdf(const hittable& _objects, const point3& _origin)
        : objects(_objects), origin(_origin)
    {
    }

    __device__ ~hittable_pdf() = default;

    __device__ float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const override;
    __device__ vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) override;

    __host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfHittable; }


private:
    const hittable& objects;
    point3 origin;
};

__device__ inline float hittable_pdf::value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const
{
    return objects.pdf_value(origin, direction, max_depth, rng);
}

__device__ inline vector3 hittable_pdf::generate(scatter_record& rec, thrust::default_random_engine& rng)
{
    return objects.random(origin, rng);
}
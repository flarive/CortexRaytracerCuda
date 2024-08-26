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

    __device__ hittable_pdf* clone() const override {
        return new hittable_pdf(*this);
    }

    __device__ ~hittable_pdf() = default;

    __device__ float value(const vector3& direction, curandState* local_rand_state) const override;
    __device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) const override;

    __device__ pdfTypeID getTypeID() const override { return pdfTypeID::pdfHittable; }


private:
    const hittable& objects;
    point3 origin;
};

__device__ inline float hittable_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
    return objects.pdf_value(origin, direction, local_rand_state);
}

__device__ inline vector3 hittable_pdf::generate(scatter_record& rec, curandState* local_rand_state) const
{
    return objects.random(origin, local_rand_state);
}
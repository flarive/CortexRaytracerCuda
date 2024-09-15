#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

#include <cmath>

class cosine_pdf : public pdf
{
public:
	__device__ cosine_pdf(const vector3& w)
	{
		m_uvw.build_from_w(w);
	}

	__device__ ~cosine_pdf() = default;

	__device__ float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const override;
	__device__ vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) override;

	__host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfCosine; }


private:
	onb m_uvw;
};


__device__ inline float cosine_pdf::value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const
{
	float cosine_theta = glm::dot(unit_vector(direction), m_uvw.w());
	return ffmax(0.0f, cosine_theta / M_PI);
}

__device__ inline vector3 cosine_pdf::generate(scatter_record& rec, thrust::default_random_engine& rng)
{
	return m_uvw.local(random_cosine_direction(rng));
}
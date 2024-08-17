#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

class cosine_pdf : public pdf
{
public:
	__host__ __device__ cosine_pdf(const vector3& w)
	{
		m_uvw.build_from_w(w);
	}

	__host__ __device__ float value(const vector3& direction, curandState* local_rand_state) const override;
	__host__ __device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) override;


private:
	onb m_uvw;
};


__host__ __device__ float cosine_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
	auto cosine_theta = dot(unit_vector(direction), m_uvw.w());
	return fmax(0, cosine_theta / M_PI);
}

__host__ __device__ vector3 cosine_pdf::generate(scatter_record& rec, curandState* local_rand_state)
{
	return m_uvw.local(random_cosine_direction(local_rand_state));
}
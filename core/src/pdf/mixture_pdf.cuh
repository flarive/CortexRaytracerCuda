#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/pdf.h
class mixture_pdf : public pdf
{
public:
	//__device__ mixture_pdf() : proportion(0.5f) { p0 = nullptr; p1 = nullptr; }
	__device__ mixture_pdf(const pdf& _p0, const pdf& _p1) : m_proportion(0.5f), m_p0(_p0), m_p1(_p1)
	{
	}
	
	__device__ mixture_pdf(const pdf& _p0, const pdf& _p1, float _prop) : m_proportion(_prop), m_p0(_p0), m_p1(_p1)
	{
	}

	__device__ mixture_pdf* clone() const override {
		return new mixture_pdf(*this);
	}

	__device__ ~mixture_pdf() = default;

	__device__ float value(const vector3& direction, curandState* local_rand_state) const override;
	__device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) const override;

	__device__ pdfTypeID getTypeID() const override { return pdfTypeID::pdfMixture; }

private:
	float m_proportion = 0.0f;
	const pdf& m_p0;
	const pdf& m_p1;
};



__device__ inline float mixture_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
	return m_proportion*(m_p0.value(direction, local_rand_state)) + (1.0f - m_proportion) * (m_p1.value(direction, local_rand_state));
}

__device__ inline vector3 mixture_pdf::generate(scatter_record& rec, curandState* local_rand_state) const
{
	if (get_real(local_rand_state) < m_proportion)
	{
		auto v0 = m_p0.generate(rec, local_rand_state);
		printf("mixture_pdf::return p[0]->generate %f %f %f\n", v0.x, v0.y, v0.z);
		return v0;
	}
	else
	{
		auto v1 = m_p1.generate(rec, local_rand_state);
		printf("mixture_pdf::return p[1]->generate %f %f %f\n", v1.x, v1.y, v1.z);
		return v1;
	}
}
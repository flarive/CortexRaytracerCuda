#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/pdf.h
class mixture_pdf : public pdf
{
public:
	__device__ mixture_pdf() : proportion(0.5) { p[0] = nullptr; p[1] = nullptr; }
	__device__ mixture_pdf(pdf* p0, pdf* p1) : proportion(0.5) { p[0] = p0; p[1] = p1; }
	__device__ mixture_pdf(pdf* p0, pdf* p1, double prop) : proportion(prop) { p[0] = p0; p[1] = p1; }

	__device__ float value(const vector3& direction, curandState* local_rand_state) const override;
	__device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) override;

public:
	double proportion = 0.0;
	pdf* p[2];
};



__device__ float mixture_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
	return proportion * (p[0]->value(direction, local_rand_state)) + (1.0 - proportion) * (p[1]->value(direction, local_rand_state));
}

__device__ vector3 mixture_pdf::generate(scatter_record& rec, curandState* local_rand_state)
{
	if (get_real(local_rand_state) < proportion)
	{
		return p[0]->generate(rec, local_rand_state);
	}
	else
	{
		return p[1]->generate(rec, local_rand_state);
	}
}

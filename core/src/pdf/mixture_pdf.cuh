#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"

// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/pdf.h
class mixture_pdf : public pdf
{
public:
	__device__ mixture_pdf() : proportion(0.5f) { p0 = nullptr; p1 = nullptr; }
	__device__ mixture_pdf(pdf* _p0, pdf* _p1) : proportion(0.5f) { p0 = _p0; p1 = _p1; }
	__device__ mixture_pdf(pdf* _p0, pdf* _p1, float _prop) : proportion(_prop) { p0 = _p0; p1 = _p1; }

	__device__ ~mixture_pdf();

	__device__ float value(const vector3& direction, curandState* local_rand_state) const override;
	__device__ vector3 generate(scatter_record& rec, curandState* local_rand_state) override;

	__host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfMixture; }

private:
	float proportion = 0.0f;
	pdf* p0 = nullptr;
	pdf* p1 = nullptr;
};



__device__ inline float mixture_pdf::value(const vector3& direction, curandState* local_rand_state) const
{
	return proportion*(p0->value(direction, local_rand_state)) + (1.0f - proportion) * (p1->value(direction, local_rand_state));
}

__device__ inline vector3 mixture_pdf::generate(scatter_record& rec, curandState* local_rand_state)
{
	//if (p[0] == nullptr || p[1] == nullptr)
	//{
	//	printf("mixture_pdf assertion failed !\n");
	//	return vector3();
	//}

	//printf("mixture_pdf::return ?\n");

	// isnan()

	if (get_real(local_rand_state) < proportion)
	{
		auto v0 = p0->generate(rec, local_rand_state);
		printf("mixture_pdf::return p[0]->generate %f %f %f\n", v0.x, v0.y, v0.z);
		return v0;
	}
	else
	{
		auto v1 = p1->generate(rec, local_rand_state);
		printf("mixture_pdf::return p[0]->generate %f %f %f\n", v1.x, v1.y, v1.z);
		return v1;
	}
}

// Destructor implementation
__device__ inline mixture_pdf::~mixture_pdf()
{
	//printf("Calling mixture_pdf destructor\n");

	if (p0) {
		delete p0;
		p0 = nullptr;
	}
	if (p1) {
		delete p1;
		p1 = nullptr;
	}
}

#pragma once

#include "../misc/vector3.cuh"
#include "../textures/image_texture.cuh"
#include "../misc/onb.cuh"
#include "../primitives/hittable.cuh"

// avoid circular dependency
struct scatter_record;


__host__ __device__ enum class pdfTypeID {
	pdfBaseType = 0,
	pdfCosine = 1,
	pdfHittable = 2,
	pdfImage = 3,
	pdfMixture = 4,
	pdfSphere = 5,
	pdfAnisotropicPhong = 6
};

/// <summary>
/// Probability Distribution Function (henceforth PDF).
/// In short, a PDF is a continuous function that can be integrated over to determine how likely a result is over an integral.
/// Remember that the PDF is a probability function.
/// </summary>
class pdf
{
public:
	__device__ virtual ~pdf() {}

	__device__ virtual float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const = 0;
	__device__ virtual vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) = 0;
};
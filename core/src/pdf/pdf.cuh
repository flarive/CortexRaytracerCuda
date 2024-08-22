#pragma once

#include "../misc/vector3.cuh"
#include "../textures/image_texture.cuh"
#include "../misc/onb.cuh"
#include "../primitives/hittable.cuh"

// avoid circular dependency
class scatter_record;


/// <summary>
/// Probability Distribution Function (henceforth PDF).
/// In short, a PDF is a continuous function that can be integrated over to determine how likely a result is over an integral.
/// Remember that the PDF is a probability function.
/// </summary>
class pdf
{
public:
	__device__ virtual ~pdf() {}

	__device__ virtual float value(const vector3& direction, curandState* local_rand_state) const = 0;
	__device__ virtual vector3 generate(scatter_record& rec, curandState* local_rand_state) = 0;
};
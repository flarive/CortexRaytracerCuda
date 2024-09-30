#pragma once

#include "primitives/hittable_list.cuh"

class device_group
{
public:
	const char* name = nullptr;
	hittable_list* value = nullptr;

	__host__ __device__ device_group()
	{

	}

	__host__ __device__ ~device_group() = default;
};
#pragma once

#include "materials/material.cuh"

class device_material
{
public:
	const char* name = nullptr;
	material* value = nullptr;
	
	__device__ device_material()
	{

	}

	__device__ ~device_material() = default;
};
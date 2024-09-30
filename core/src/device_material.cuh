#pragma once

#include "materials/material.cuh"

class device_material
{
public:
	const char* name = nullptr;
	material* value = nullptr;
	
	__host__ __device__ device_material()
	{
	}

	__host__ __device__ device_material(const char* _name, material* _value)
		: name(_name), value(_value)
	{
	}

	__host__ __device__ ~device_material() = default;
};
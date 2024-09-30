#pragma once

#include "textures/texture.cuh"

class device_texture
{
public:
	const char* name = nullptr;
	texture* value = nullptr;
	
	__host__ __device__ device_texture()
	{

	}

	__host__ __device__ ~device_texture() = default;
};
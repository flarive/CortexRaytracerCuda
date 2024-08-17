#pragma once

#include "renderer.h"

#include "../misc/render_parameters.h"
#include "../cameras/camera.cuh"
#include "../samplers/sampler.cuh"

class gpu_cuda_renderer : public renderer
{
public:
	void render(scene& _scene, camera& _camera, const render_parameters& _params, sampler* aa_sampler) const;
};


// Declaration of the CUDA kernel launcher
extern void launchGPU(int nx, int ny, int ns, int tx, int ty, const char* filepath, bool quietMode);


void gpu_cuda_renderer::render(scene& _scene, camera& camera, const render_parameters& _params, sampler* aa_sampler) const
{
	if (!_params.quietMode)
		std::clog << "Using GPU Cuda renderer" << std::endl;

	std::cout << "[INFO] Starting GPU Cuda rendering" << std::endl;

	launchGPU(_params.width, _params.height, _params.samplePerPixel, 16, 16, _params.saveFilePath.c_str(), true);
}
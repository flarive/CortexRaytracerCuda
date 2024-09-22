#pragma once

#include "renderer.h"

#include "../misc/render_parameters.h"
#include "../cameras/camera.cuh"
#include "../samplers/sampler.cuh"

class gpu_cuda_renderer : public renderer
{
public:
	void render(const sceneConfig& _sceneCfg, const render_parameters& _params) const;
};


// Declaration of the CUDA kernel launcher
extern void launchGPU(const sceneConfig& _sceneCfg, int width, int height, int spp, int max_depth, int tx, int ty, const char* filepath, bool quietMode);


void gpu_cuda_renderer::render(const sceneConfig& _sceneCfg, const render_parameters& _params) const
{
	if (!_params.quietMode)
		std::clog << "Using GPU Cuda renderer" << std::endl;

	std::cout << "[INFO] Starting GPU Cuda rendering" << std::endl;

	launchGPU(_sceneCfg, _params.width, _params.height, _params.samplePerPixel, _params.recursionMaxDepth, 16, 16, _params.saveFilePath.c_str(), true);
}
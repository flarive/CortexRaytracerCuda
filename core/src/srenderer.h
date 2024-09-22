#pragma once

#include "misc/scene.cuh"
#include "misc/render_parameters.h"
#include "renderers/renderer.h"
#include "renderers/gpu_cuda_renderer.h"
#include "cameras/perspective_camera.cuh"
#include "scenes/scene_config.h"

class srenderer
{
public:

	void render(const sceneConfig& _sceneCfg, const render_parameters& _params);
};


void srenderer::render(const sceneConfig& _sceneCfg, const render_parameters& _params)
{
    //std::cout << "[INFO] Init scene" << std::endl;

    //camera* cam = _scene.get_camera();

    //_scene.extract_emissive_objects();

    //std::cout << "[INFO] Optimizing scene" << std::endl;

    //int seed = 78411111;
    //thrust::minstd_rand rng(seed);
    //thrust::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

    //_scene.build_optimized_world(rng);


    // init default anti aliasing sampler
    //auto sampler = new random_sampler(camera->get_pixel_delta_u(), camera->get_pixel_delta_v(), camera->getSamplePerPixel());



    std::unique_ptr<renderer> r;


    //if (!_params.use_gpu)
    //{
    //    // cpu
    //    if (_params.use_multi_thread)
    //    {
    //        r = std::make_unique<cpu_multithread_renderer>();
    //    }
    //    else
    //    {
    //        r = std::make_unique<cpu_singlethread_renderer>();
    //    }
    //}
    //else
    //{
    //    // gpu
        r = std::make_unique<gpu_cuda_renderer>();
    //}

    if (r)
        r->render(_sceneCfg, _params);
}
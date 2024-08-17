#pragma once

#include "misc/scene.h"
#include "misc/render_parameters.h"
#include "renderers/renderer.h"
#include "renderers/gpu_cuda_renderer.h"

class srenderer
{
public:

	void render(scene& _scene, const render_parameters& _params);
};


void srenderer::render(scene& _scene, const render_parameters& _params)
{
    std::cout << "[INFO] Init scene" << std::endl;

    camera *cam = new camera(
        vector3(0, 1, 0),
        vector3(0, 1, 0),
        vector3(0, 1, 0),
        40.0,
        float(_params.width) / float(_params.height),
        1.0,
        2.0,
        0.0,
        1.0
    );

    //camera* camera = _scene.get_camera();
    //camera->initialize(_params);

    //_scene.extract_emissive_objects();

    std::cout << "[INFO] Optimizing scene" << std::endl;

    //_scene.build_optimized_world();


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
        r->render(_scene, *cam, _params, nullptr);
}
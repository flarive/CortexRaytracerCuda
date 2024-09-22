#include "misc/render_parameters.h"
#include "misc/scene.cuh"
#include "misc/timer.h"
#include "srenderer.h"
#include "scenes/scene_builder.h"
#include "scenes/scene_config.h"
#include "scenes/scene_manager.h"


int main(int argc, char* argv[])
{
	render_parameters params = render_parameters::getArgs(argc, argv);

    // Create world
    scene_manager builder;

    std::cout << "[INFO] Ready !" << std::endl;

    timer renderTimer;

    sceneConfig sceneCfg = builder.load_scene(params);

    // Start measuring time
    renderTimer.start();

    srenderer render;
    render.render(sceneCfg, params);

    // Stop measuring time
    renderTimer.stop();

    if (!params.quietMode)
        renderTimer.displayTime();

    std::cout << "[INFO] Finished !" << std::endl;

    //std::string dummy;
    //std::cout << "Enter to continue..." << std::endl;
    //std::getline(std::cin, dummy);

    exit(EXIT_SUCCESS);
}
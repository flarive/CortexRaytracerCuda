#include "misc/render_parameters.h"
#include "misc/scene.h"
#include "misc/timer.h"
#include "srenderer.h"


int main(int argc, char* argv[])
{
	render_parameters params = render_parameters::getArgs(argc, argv);


    std::cout << "[INFO] Ready !" << std::endl;

    timer renderTimer;

    scene world;

    // Start measuring time
    renderTimer.start();

    srenderer render;
    render.render(world, params);

    // Stop measuring time
    renderTimer.stop();

    if (!params.quietMode)
        renderTimer.displayTime();

    std::cout << "[INFO] Finished !" << std::endl;

    exit(EXIT_SUCCESS);
}
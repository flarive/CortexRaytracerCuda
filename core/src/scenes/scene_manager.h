#pragma once

#include "scene_config.h"
#include "../misc/render_parameters.h"

class scene_manager
{
public:
    sceneConfig load_scene(const render_parameters& params);
};
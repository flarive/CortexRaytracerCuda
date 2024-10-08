#include "scene_manager.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <stb/stb_image.h>

#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"
#include "../misc/scene.cuh"

#include "../primitives/aarect.cuh"
#include "../primitives/box.cuh"
#include "../primitives/sphere.cuh"
#include "../primitives/quad.cuh"
#include "../primitives/cylinder.cuh"
#include "../primitives/cone.cuh"
#include "../primitives/disk.cuh"
#include "../primitives/torus.cuh"
#include "../primitives/volume.cuh"

#include "../primitives/translate.cuh"
#include "../primitives/scale.cuh"
#include "../primitives/rotate.cuh"

#include "../lights/omni_light.cuh"
#include "../lights/directional_light.cuh"

#include "../cameras/perspective_camera.cuh"
#include "../cameras/orthographic_camera.cuh"

#include "../materials/material.cuh"
#include "../materials/lambertian.cuh"
#include "../materials/metal.cuh"
#include "../materials/dielectric.cuh"
#include "../materials/phong.cuh"
#include "../materials/oren_nayar.cuh"
#include "../materials/isotropic.cuh"
#include "../materials/anisotropic.cuh"
#include "../materials/diffuse_light.cuh"

#include "../utilities/uvmapping.cuh"

#include "../textures/solid_color_texture.cuh"
#include "../textures/checker_texture.cuh"
#include "../textures/image_texture.cuh"
#include "../textures/perlin_noise_texture.cuh"
#include "../textures/gradient_texture.cuh"
#include "../textures/alpha_texture.cuh"
#include "../textures/bump_texture.cuh"
#include "../textures/normal_texture.cuh"

#include "../pdf/image_pdf.cuh"

#include "../misc/bvh_node.cuh"

#include "scene_loader.h"
#include "scene_builder.h"
#include "scene_config.h"



sceneConfig scene_manager::load_scene(const render_parameters& params)
{
    sceneConfig fullCfg;

    // get data from .scene file
    scene_loader config(params.sceneName);
    scene_builder scene = config.loadSceneFromFile();


    fullCfg.imageCfg = scene.getImageConfig();
    fullCfg.cameraCfg = scene.getCameraConfig();
    fullCfg.lightsCfg = scene.getLightsConfig();
    fullCfg.texturesCfg = scene.getTexturesConfig();
    fullCfg.materialsCfg = scene.getMaterialsConfig();
    fullCfg.primitivesCfg = scene.getPrimitivesConfig();
    fullCfg.meshesCfg = scene.getMeshesConfig();


    // command line parameters are stronger than .scene parameters
    fullCfg.cameraCfg.aspectRatio = params.ratio;
    fullCfg.imageCfg.width = params.width;
    fullCfg.imageCfg.spp = params.samplePerPixel; // antialiasing quality
    fullCfg.imageCfg.depth = params.recursionMaxDepth; // max nbr of bounces a ray can do

    return fullCfg;
}
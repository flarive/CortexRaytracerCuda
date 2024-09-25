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
//#include "../utilities/mesh_loader.cuh"

#include "../textures/solid_color_texture.cuh"
#include "../textures/checker_texture.cuh"
#include "../textures/image_texture.cuh"
#include "../textures/perlin_noise_texture.cuh"
#include "../textures/gradient_texture.cuh"
#include "../textures/alpha_texture.cuh"
#include "../textures/bump_texture.cuh"
//#include "../textures/roughness_texture.cuh"
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




    // ?????????????????????????????????????????????

    //auto zzz = scene.getSceneObjects();
    //world.set(zzz.objects, zzz.object_count);

    //camera* cam = nullptr;



    //if (!cameraCfg.isOrthographic)
    //{
    //    cam = new perspective_camera();
    //    cam->vfov = cameraCfg.fov;
    //}
    //else
    //{
    //    cam = new orthographic_camera();
    //    cam->ortho_height = cameraCfg.orthoHeight;
    //}


    //cam->aspect_ratio = cameraCfg.aspectRatio;
    //cam->image_width = imageCfg.width;
    //cam->samples_per_pixel = imageCfg.spp; // denoiser quality
    //cam->max_depth = imageCfg.depth; // max nbr of bounces a ray can do
    //cam->background_color = color(0.70f, 0.80f, 1.00f);
    //cam->lookfrom = cameraCfg.lookFrom;
    //cam->lookat = cameraCfg.lookAt;
    //cam->vup = cameraCfg.upAxis;
    //cam->is_orthographic = cameraCfg.isOrthographic;

    //
    //// Background
    //if (imageCfg.background.filepath)
    //{
    //    //auto background = new image_texture(imageCfg.background.filepath);
    //    //cam->background_texture = background;
    //    //cam->background_iskybox = imageCfg.background.is_skybox;

    //    //if (imageCfg.background.is_skybox)
    //    //    cam->background_pdf = new image_pdf(background);
    //}
    //else
    //{
    //    cam->background_color = imageCfg.background.rgb;
    //}



    //// command line parameters are stronger than .scene parameters
    //cam->aspect_ratio = params.ratio;
    //cam->image_width = params.width;
    //cam->samples_per_pixel = params.samplePerPixel; // antialiasing quality
    //cam->max_depth = params.recursionMaxDepth; // max nbr of bounces a ray can do


    //// Depth of field
    //cam->defocus_angle = cameraCfg.aperture;
    //cam->focus_dist = cameraCfg.focus;

    //world.set_camera(cam);

    return fullCfg;
}
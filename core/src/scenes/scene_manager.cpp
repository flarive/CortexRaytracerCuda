//#include "scene_manager.h"
//
//#include <glm/glm.hpp>
//#include <glm/gtx/transform.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//
//#include <stb/stb_image.h>
//
//#include "../primitives/hittable.cuh"
//#include "../primitives/hittable_list.cuh"
//#include "../misc/scene.cuh"
//
//#include "../primitives/aarect.cuh"
//#include "../primitives/box.cuh"
//#include "../primitives/sphere.cuh"
//#include "../primitives/quad.cuh"
//#include "../primitives/cylinder.h"
//#include "../primitives/cone.h"
//#include "../primitives/disk.h"
//#include "../primitives/torus.h"
//#include "../primitives/volume.cuh"
//
//#include "../primitives/translate.cuh"
//#include "../primitives/scale.cuh"
//#include "../primitives/rotate.cuh"
//
//#include "../lights/omni_light.h"
//#include "../lights/directional_light.h"
//
//#include "../cameras/perspective_camera.cuh"
//
//#include "../materials/material.cuh"
//#include "../materials/lambertian.cuh"
//#include "../materials/metal.cuh"
//#include "../materials/dielectric.cuh"
//#include "../materials/phong.h"
//#include "../materials/oren_nayar.h"
//#include "../materials/isotropic.h"
//#include "../materials/anisotropic.h"
//#include "../materials/diffuse_light.cuh"
//
//#include "../utilities/uvmapping.cuh"
//#include "../utilities/mesh_loader.h"
//
//#include "../textures/solid_color_texture.cuh"
//#include "../textures/checker_texture.cuh"
//#include "../textures/image_texture.cuh"
//#include "../textures/perlin_noise_texture.cuh"
//#include "../textures/gradient_texture.h"
//#include "../textures/alpha_texture.h"
//#include "../textures/bump_texture.h"
//#include "../textures/roughness_texture.h"
//#include "../textures/normal_texture.h"
//
//#include "../pdf/image_pdf.cuh"
//
//#include "../misc/bvh_node.cuh"
//
//#include "../misc/aabb_debug.h"
//
//#include "scene_loader.h"
//#include "scene_builder.h"
//
//
//
//
//scene scene_manager::random_spheres(perspective_camera &cam)
//{
//    scene world;
//
//    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
//    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));
//
//    for (int a = -11; a < 11; a++)
//    {
//        for (int b = -11; b < 11; b++)
//        {
//            auto choose_mat = randomizer::random_double();
//            point3 center(a + 0.9 * randomizer::random_double(), 0.2, b + 0.9 * randomizer::random_double());
//
//            if ((center - point3(4, 0.2, 0)).length() > 0.9)
//            {
//                shared_ptr<material> sphere_material;
//
//                if (choose_mat < 0.8)
//                {
//                    // diffuse
//                    auto albedo = color::random() * color::random();
//                    sphere_material = make_shared<lambertian>(albedo);
//                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
//
//                    //std::cout << "{" << std::endl;
//                    //std::cout << "  name = \"Sphere-" << a << "-" << b << "\";" << std::endl;
//                    //std::cout << "  position = { x = " << center.x << "; y = " << center.y << "; z = " << center.z << "; };" << std::endl;
//                    //std::cout << "  radius = 0.2;" << std::endl;
//                    //std::cout << "  material = \"sphere_material_color_" << a << "_" << b << "\";" << std::endl;
//                    //std::cout << "}," << std::endl;
//
//                    //std::cout << "{" << std::endl;
//                    //std::cout << "  name = \"sphere_material_color_" << a << "_" << b << "\";" << std::endl;
//                    //std::cout << "  color = { r = " << albedo.r() << "; g = " << albedo.g() << "; b = " << albedo.b() << "; };" << std::endl;
//                    //std::cout << "}," << std::endl;
//
//                    //{
//                    //    name = "sphere_material_color";
//                    //    texture = "material2_texture";
//                    //}
//                }
//                else if (choose_mat < 0.95)
//                {
//                    // metal
//                    auto albedo = color::random(0.5, 1);
//                    auto fuzz = randomizer::random_double(0, 0.5);
//                    sphere_material = make_shared<metal>(albedo, fuzz);
//                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
//
//                    std::cout << "{" << std::endl;
//                    std::cout << "  name = \"sphere_material_metal_" << a << "_" << b << "\";" << std::endl;
//                    std::cout << "  color = { r = " << albedo.r() << "; g = " << albedo.g() << "; b = " << albedo.b() << "; };" << std::endl;
//                    std::cout << "  fuzz = " << fuzz << "; };" << std::endl;
//                    std::cout << "}," << std::endl;
//
//                    //std::cout << "{" << std::endl;
//                    //std::cout << "  name = \"Sphere-" << a << "-" << b << "\";" << std::endl;
//                    //std::cout << "  position = { x = " << center.x << "; y = " << center.y << "; z = " << center.z << "; };" << std::endl;
//                    //std::cout << "  radius = 0.2;" << std::endl;
//                    //std::cout << "  material = \"sphere_material_metal_" << a << "_" << b << "\";" << std::endl;
//                    //std::cout << "}," << std::endl;
//                }
//                else
//                {
//                    // glass
//                    sphere_material = make_shared<dielectric>(1.5);
//                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
//
//                    //std::cout << "{" << std::endl;
//                    //std::cout << "  name = \"Sphere-" << a << "-" << b << "\";" << std::endl;
//                    //std::cout << "  position = { x = " << center.x << "; y = " << center.y << "; z = " << center.z << "; };" << std::endl;
//                    //std::cout << "  radius = 0.2;" << std::endl;
//                    //std::cout << "  material = \"sphere_material_glass\";" << std::endl;
//                    //std::cout << "}," << std::endl;
//                }
//            }
//        }
//    }
//
//    auto material1 = make_shared<dielectric>(1.5);
//    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));
//
//    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
//    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));
//
//    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
//    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));
//
//    cam.vfov = 20; // vertical field of view
//    cam.lookfrom = point3(13, 2, 3); // camera position in world
//    cam.lookat = point3(0, 0, 0); // camera target in world
//
//    cam.defocus_angle = 0.6; // depth-of-field large aperture
//    cam.focus_dist = 10.0; // depth-of-field large aperture
//
//    return world;
//}
//
//scene scene_manager::two_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto checker_material = make_shared<checker_texture>(0.8, color(0,0,0), color(1,1,1));
//
//    world.add(make_shared<sphere>(point3(0, -10, 0), 10, make_shared<lambertian>(checker_material)));
//    world.add(make_shared<sphere>(point3(0, 10, 0), 10, make_shared<lambertian>(checker_material)));
//
//
//    cam.vfov = 20;
//    cam.lookfrom = point3(13, 2, 3);
//    cam.lookat = point3(0, 0, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::two_perlin_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto pertext = make_shared<perlin_noise_texture>(4);
//    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
//    world.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));
//
//    cam.vfov = 20;
//    cam.lookfrom = point3(13, 2, 3);
//    cam.lookat = point3(0, 0, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//
//scene scene_manager::advanced_lights(perspective_camera& cam)
//{
//    scene world;
//
//    auto pertext = make_shared<perlin_noise_texture>(4);
//    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, make_shared<lambertian>(pertext)));
//    world.add(make_shared<sphere>(point3(0, 2, 0), 2, make_shared<lambertian>(pertext)));
//
//    // Box
//    auto red = make_shared<lambertian>(color(.65, .05, .05));
//    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(3, 3, 3), red);
//    box1 = make_shared<rt::rotate>(box1, vector3(0, 15, 0));
//    box1 = make_shared<rt::translate>(box1, vector3(-10, 0, 5));
//    world.add(box1);
//
//    // Light Sources
//	//auto light1 = make_shared<directional_light>(point3(3, 1, -2), vector3(2, 0, 0), vector3(0, 2, 0), 2, color(10, 10, 10), "QuadLight1");
// //   world.add(light1);
//
//	auto light2 = make_shared<omni_light>(point3(0, 7, 0), 1, 0.0, color(0.0, 0.0, 0.0), "SphereLight2");
//	world.add(light2);
//        
//    cam.vfov = 26;
//    cam.lookfrom = point3(26, 3, 6);
//    cam.lookat = point3(0, 2, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color(0.0, 0.0, 0.0);
//
//    return world;
//}
//
//scene scene_manager::cornell_box(perspective_camera& cam)
//{
//    scene world;
//
//    auto red = make_shared<lambertian>(color(.65, .05, .05));
//    auto white = make_shared<lambertian>(color(.73, .73, .73));
//    auto green = make_shared<lambertian>(color(.12, .45, .15));
//    auto light = make_shared<diffuse_light>(color(15, 15, 15));
//
//    shared_ptr<material> aluminum = make_shared<metal>(color(0.8, 0.85, 0.88), 0.0);
//    shared_ptr<dielectric> glass = make_shared<dielectric>(1.5);
//
//    // Cornell box sides
//    //world.add(make_shared<quad>(point3(555, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), green));
//
//    //world.add(make_shared<quad>(point3(0, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), red));
//
//    //world.add(make_shared<quad>(point3(0, 0, 0), vector3(555, 0, 0), vector3(0, 0, 555), white));
//
//    //world.add(make_shared<quad>(point3(555, 555, 555), vector3(-555, 0, 0), vector3(0, 0, -555), white));
//
//    world.add(make_shared<quad>(point3(0, 0, 555), vector3(555, 0, 0), vector3(0, 555, 0), white));
//
//    /// Light
//    //world.add(make_shared<quad>(point3(213, 554, 227), vector3(130, 0, 0), vector3(0, 0, 105), light));
//
//    // Aluminium Box
//    //shared_ptr<hittable> box1 = make_shared<box>(point3(0, 165, 0), point3(165, 330, 165), aluminum);
//    //box1 = make_shared<rt::rotate>(box1, vector3(0, 15, 0));
//    //box1 = make_shared<rt::translate>(box1, vector3(310, 0, 295));
//    //world.add(box1);
//
//    //// Glass Sphere
//    //world.add(make_shared<sphere>(point3(190, 90, 190), 90, glass));
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(278, 554, 332), vector3(-130, 0, 0), vector3(0, 0, -105), 1.5, color(15, 15, 15), "QuadLight1"));
//
//    
//
//
//    cam.vfov = 40;
//    cam.lookfrom = point3(278, 278, -800);
//    cam.lookat = point3(278, 278, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color(0, 0, 0);
//
//    world.set_camera(std::make_shared<perspective_camera>(cam));
//
//    return world;
//}
//
//scene scene_manager::cornell_box_custom(perspective_camera& cam)
//{
//    scene world;
//
//    auto red = make_shared<lambertian>(color(.65, .05, .05));
//    auto white = make_shared<lambertian>(color(.73, .73, .73));
//    auto green = make_shared<lambertian>(color(.12, .45, .15));
//    auto light = make_shared<diffuse_light>(color(15, 15, 15));
//
//    shared_ptr<material> aluminum = make_shared<metal>(color(0.8, 0.85, 0.88), 0.0);
//    shared_ptr<dielectric> glass = make_shared<dielectric>(1.5);
//
//    // Cornell box sides
//    world.add(make_shared<quad>(point3(555, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), green));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), red));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(555, 0, 0), vector3(0, 0, 555), white));
//    world.add(make_shared<quad>(point3(555, 555, 555), vector3(-555, 0, 0), vector3(0, 0, -555), white));
//    world.add(make_shared<quad>(point3(0, 0, 555), vector3(555, 0, 0), vector3(0, 555, 0), white));
//
//    // Aluminium Box
//    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 165, 0), point3(165, 330, 165), aluminum, "AluBox");
//    box1 = make_shared<rt::rotate>(box1, vector3(0, 15, 0));
//    box1 = make_shared<rt::translate>(box1, vector3(265, 0, 295));
//    world.add(box1);
//
//    // Glass Sphere
//    world.add(make_shared<sphere>(point3(190, 90, 190), 90, glass, "GlassSphere"));
//
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(343, 554, 332), vector3(-130, 0, 0), vector3(0, 0, -105), 1.5, color(15, 15, 15), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(343 - 65, 450, 332), 65, 1.0, color(4, 4, 4), "SphereLight2", false));
//
//    cam.vfov = 40;
//    cam.lookfrom = point3(278, 278, -800);
//    cam.lookat = point3(278, 278, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color(0, 0, 0);
//
//    return world;
//}
//
//scene scene_manager::cornell_box_smoke(perspective_camera& cam)
//{
//    scene world;
//
//    auto red = make_shared<lambertian>(color(.65, .05, .05));
//    auto white = make_shared<lambertian>(color(.73, .73, .73));
//    auto green = make_shared<lambertian>(color(.12, .45, .15));
//
//    world.add(make_shared<quad>(point3(555, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), green));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), red));
//    world.add(make_shared<quad>(point3(0, 555, 0), vector3(555, 0, 0), vector3(0, 0, 555), white));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(555, 0, 0), vector3(0, 0, 555), white));
//    world.add(make_shared<quad>(point3(0, 0, 555), vector3(555, 0, 0), vector3(0, 555, 0), white));
//
//
//    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), white);
//    box1 = make_shared<rt::rotate>(box1, vector3(0, 15, 0));
//    box1 = make_shared<rt::translate>(box1, vector3(265, 0, 295));
//
//    shared_ptr<hittable> box2 = make_shared<box>(point3(0, 0, 0), point3(165, 165, 165), white);
//    box2 = make_shared<rt::rotate>(box2, vector3(0, -18, 0));
//    box2 = make_shared<rt::translate>(box2, vector3(130, 0, 65));
//
//    world.add(make_shared<volume>(box1, 0.01, color(0, 0, 0)));
//    world.add(make_shared<volume>(box2, 0.01, color(1, 1, 1)));
//
//    // Light Sources
//    auto light1 = make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.5, color(5, 5, 5), "QuadLight1");
//    world.add(light1);
//
//
//
//    cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 40;
//    cam.lookfrom = point3(278, 278, -800);
//    cam.lookat = point3(278, 278, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::cornell_box_phong(perspective_camera& cam)
//{
//    scene world;
//
//    auto red = make_shared<lambertian>(color(.65, .05, .05));
//    auto white = make_shared<lambertian>(color(.73, .73, .73));
//    auto green = make_shared<lambertian>(color(.12, .45, .15));
//    auto light = make_shared<diffuse_light>(color(15, 15, 15));
//
//    shared_ptr<material> aluminum = make_shared<metal>(color(0.8, 0.85, 0.88), 0.0);
//    shared_ptr<dielectric> glass = make_shared<dielectric>(1.5);
//
//    // Cornell box sides
//    world.add(make_shared<quad>(point3(555, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), green));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(0, 555, 0), vector3(0, 0, 555), red));
//    world.add(make_shared<quad>(point3(0, 0, 0), vector3(555, 0, 0), vector3(0, 0, 555), white));
//    world.add(make_shared<quad>(point3(555, 555, 555), vector3(-555, 0, 0), vector3(0, 0, -555), white));
//    world.add(make_shared<quad>(point3(0, 0, 555), vector3(555, 0, 0), vector3(0, 555, 0), white));
//
//    // Aluminium Box
//    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), aluminum, "AluBox");
//    box1 = make_shared<rt::rotate>(box1, vector3(0, 20, 0));
//    box1 = make_shared<rt::translate>(box1, vector3(265, 0, 295));
//    world.add(box1);
//
//    double ambient = 0.1;
//    double diffuse = 0.9;
//    double specular = 0.0;
//    double shininess = 0.0;
//
//
//    auto phong_material = make_shared<phong>(color(0.8, 0.1, 0.2), ambient, diffuse, specular, shininess);
//
//
//
//    // Phong Sphere
//    world.add(make_shared<sphere>(point3(190, 90, 190), 90, phong_material, "PhongSphere"));
//
//
//
//    // Light Sources
//    shared_ptr<hittable> light1 = make_shared<directional_light>(point3(343, 554, 332), vector3(-130, 0, 0), vector3(0, 0, -105), 5, color(2,2,2), "QuadLight1");
//    world.add(light1);
//
//    //auto light2 = make_shared<omni_light>(point3(343 - 65, 450, 332), 65, 1.0, color(4, 4, 4), "SphereLight2", false);
//    //world.add(light2);
//
//    cam.vfov = 40;
//    cam.lookfrom = point3(278, 278, -800);
//    cam.lookat = point3(278, 278, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color(0, 0, 0);
//
//    return world;
//}
//
//scene scene_manager::final_scene(perspective_camera& cam)
//{
//    hittable_list boxes1;
//    auto ground = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//
//    int boxes_per_side = 20;
//    for (int i = 0; i < boxes_per_side; i++) {
//        for (int j = 0; j < boxes_per_side; j++) {
//            auto w = 100.0;
//            auto x0 = -1000.0 + i * w;
//            auto z0 = -1000.0 + j * w;
//            auto y0 = 0.0;
//            auto x1 = x0 + w;
//            auto y1 = randomizer::random_double(1, 101);
//            auto z1 = z0 + w;
//
//            //std::cout << "{" << std::endl;
//            //std::cout << "  name = \"Box-" << i << "-" << j << "\";" << std::endl;
//            //std::cout << "  position = { x = " << x0 << "; y = " << (y0 + (y1 / 2.0)) << "; z = " << z0 << "; };" << std::endl;
//            //std::cout << "  size = { x = " << 100 << "; y = " << 100 << "; z = " << 100 << "; };" << std::endl;
//            //std::cout << "  material = \"ground_material\";" << std::endl;
//            //std::cout << "  group = \"Boxes1\";" << std::endl;
//            //std::cout << "}," << std::endl;
//
//
//            //{
//            //    name = "Box1";
//            //    position = { x = 0.0; y = 0.0; z = 0.0; };
//            //    size = { x = 165.0; y = 330.0; z = 165.0; };
//            //    material = "white_material";
//            //},
//
//
//            boxes1.add(make_shared<box>(point3(x0, y0 + (y1 / 2.0), z0), point3(100, 100, 100), ground));
//        }
//    }
//
//    scene world;
//
//    world.add(make_shared<bvh_node>(boxes1));
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(123, 554, 147), vector3(300, 0, 0), vector3(0, 0, 265), 1.5, color(7, 7, 7), "QuadLight1", false));
//
//
//
//    auto center1 = point3(400, 400, 200);
//    auto center2 = center1 + vector3(30, 0, 0);
//
//    auto sphere_material = make_shared<lambertian>(color(0.7, 0.3, 0.1));
//    world.add(make_shared<sphere>(center1, center2, 50, sphere_material));
//
//    world.add(make_shared<sphere>(point3(260, 150, 45), 50, make_shared<dielectric>(1.5)));
//    world.add(make_shared<sphere>(point3(0, 150, 145), 50, make_shared<metal>(color(0.8, 0.8, 0.9), 1.0)));
//
//
//
//
//    //auto boundary = make_shared<sphere>(point3(360, 150, 145), 70, make_shared<dielectric>(1.5));
//    //world.add(boundary);
//
//
//    //world.add(make_shared<volume>(boundary, 0.2, color(0.2, 0.4, 0.9)));
//
//
//
//    //boundary = make_shared<sphere>(point3(0, 0, 0), 5000, make_shared<dielectric>(1.5));
//    //world.add(make_shared<volume>(boundary, .0001, color(1, 1, 1)));
//
//
//
//
//    auto emat = make_shared<lambertian>(make_shared<image_texture>("../../data/textures/earthmap.jpg"));
//    world.add(make_shared<sphere>(point3(400, 200, 400), 100, emat));
//
//
//    auto pertext = make_shared<perlin_noise_texture>(0.1);
//    world.add(make_shared<sphere>(point3(220, 280, 300), 80, make_shared<lambertian>(pertext)));
//
//    hittable_list boxes2;
//    auto white = make_shared<lambertian>(color(.73, .73, .73));
//    int ns = 1000;
//    for (int j = 0; j < ns; j++) {
//        
//        auto v = randomizer::random_vector(0, 165);
//        
//        boxes2.add(make_shared<sphere>(v, 10, white));
//
//        //std::cout << "{" << std::endl;
//        //std::cout << "  name = \"Sphere-" << j << "\";" << std::endl;
//        //std::cout << "  position = { x = " << v.x << "; y = " << v.y << "; z = " << v.z << "; };" << std::endl;
//        //std::cout << "  radius = 10.0;" << std::endl;
//        //std::cout << "  material = \"white_material\";" << std::endl;
//        //std::cout << "  group = \"Boxes2\";" << std::endl;
//        //std::cout << "}," << std::endl;
//
//    }
//
//    world.add(make_shared<rt::translate>(make_shared<rt::rotate>(make_shared<bvh_node>(boxes2), vector3(0, 15, 0)), vector3(-100, 270, 395)));
//
//
//    cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 40;
//    cam.lookfrom = point3(478, 278, -600);
//    cam.lookat = point3(278, 278, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::cow_scene(perspective_camera& cam)
//{
//    scene world;
//
//    // Materials
//    auto diffuseRed = make_shared<lambertian>(color(0.8, 0.1, 0.1));
//    auto diffuseGrey = make_shared<lambertian>(color(0.5, 0.5, 0.5));
//    auto diffuseBlue = make_shared<lambertian>(color(0.1, 0.1, 0.9));
//    auto uvmapper_material = make_shared<lambertian>(make_shared<image_texture>("../../data/textures/uv_mapper_no_numbers.jpg"));
//
//
//    //auto floor = rtw_stb_obj_loader::load_model_from_file("../../data/models/floor_big.obj", diffuseRed, false);
//    //floor = make_shared<rt::scale>(floor, vector3(1.0, 1.0, 1.0));
//    //floor = make_shared<rt::translate>(floor, vector3(0.0, -3.0, 0.0));
//    ////cow = make_shared<raytracer::rotate>(floor, 90, 1);
//    //world.add(floor);
//
//
//    //// Load mesh
//    //auto cow = rtw_stb_obj_loader::load_model_from_file("../../data/models/cow.obj", diffuseGrey, true);
//    //cow = make_shared<rt::scale>(cow, vector3(1.0, 1.0, 1.0));
//    //cow = make_shared<rt::translate>(cow, vector3(0.0, 0.6, 0.0));
//    ////cow = make_shared<raytracer::rotate>(cow, 0, 1);
//    //world.add(cow);
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 2.0, color(4, 4, 4), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0, 50, 332), 65, 1.0, color(4, 4, 4), "SphereLight2", false));
//
//
//    cam.vfov = 22;
//    cam.lookfrom = point3(0, 2, 25);
//    cam.lookat = point3(0, 0, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::nautilus_scene(perspective_camera& cam)
//{
//    scene world;
//
//    // Materials
//    auto diffuseRed = make_shared<lambertian>(color(0.8, 0.1, 0.1));
//    auto diffuseGrey = make_shared<lambertian>(color(0.5, 0.5, 0.5));
//
//    auto nautilus_texture = make_shared<phong>(make_shared<image_texture>("../../data/models/nautilus_diffuse.jpg"), 0.1, 0.9, 0.8, 0.1);
//
//    // Load mesh
//	//auto nautilus = rtw_stb_obj_loader::load_model_from_file("../../data/models/nautilus.obj", nautilus_texture, false, true);
//	//nautilus = make_shared<rt::scale>(nautilus, vector3(0.05, 0.05, 0.05));
//	//nautilus = make_shared<rt::translate>(nautilus, vector3(0, -3, 0));
//	//nautilus = make_shared<rt::rotate>(nautilus, vector3(0, 90, 0));
//	//world.add(nautilus);
//
//    // Debug
//    //world.add(aabb_debug::aabb_to_box_primitive(nautilus->bounding_box()));
//
//
//
//	// Light Sources
//	world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.0, color(4, 4, 4), "QuadLight1"));
//	//world.add(make_shared<omni_light>(point3(0, 50, 332), 65, 1.0, color(4, 4, 4), "SphereLight2", false));
//
//
//    cam.vfov = 22;
//    cam.lookfrom = point3(0, 10, 25);
//    cam.lookat = point3(0, 0, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color::black();
//
//    return world;
//}
//
//scene scene_manager::extended_primitives(perspective_camera& cam)
//{
//    scene world;
//
//    auto ground_material = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//
//
//    //auto lambertian_material = make_shared<lambertian>(color(0.1, 0.2, 0.9));
//
//    auto uvmapper_material = make_shared<lambertian>(make_shared<image_texture>("../../data/textures/uv_mapper_no_numbers.jpg"));
//
//
//    world.add(make_shared<quad>(point3(-6, 0, 5), vector3(12, 0, 0), vector3(0, 0, -12), ground_material));
//
//    // Cylinder
//    world.add(make_shared<cylinder>(point3(-2.0, 0.0, 0.0), 0.4, 1.0, uvmapper_material, uvmapping(1.0, 1.0, 0, 0)));
//    world.add(make_shared<disk>(point3(-2.0, 0.5, 0.0), 0.4, 0.2, uvmapper_material, uvmapping(1.0, 1.0, 0, 0)));
//
//    // Cone
//    shared_ptr<hittable> cone1 = make_shared<cone>(point3(-1.0, 0.0, 0.0), 0.4, 1.0, uvmapper_material, uvmapping(1.0, 1.0, 0, 0));
//    //cone1 = make_shared<raytracer::scale>(cone1, 1,1,1);
//    //cone1 = make_shared<raytracer::translate>(cone1, vector3(1,0,0));
//    //cone1 = make_shared<raytracer::rotate>(cone1, 45, 0);
//    world.add(cone1);
//
//    // Box
//    shared_ptr<hittable> box1 = make_shared<box>(point3(0.0, 0.35, 0.0), point3(0.7, 0.7, 0.7), uvmapper_material, uvmapping(0.5, 0.5, 0, 0));
//    //box1 = make_shared<raytracer::rotate>(box1, 30, 1);
//    //box1 = make_shared<raytracer::translate>(box1, vector3(-0.5, 0, 0));
//    //box1 = make_shared<rt::scale>(box1, vector3(0.2, 0.2, 0.2));
//    world.add(box1);
//
//    // Torus
//    shared_ptr<hittable> torus1 = make_shared<torus>(point3(1.0, 0.4, 0.0), 0.3, 0.15, uvmapper_material, uvmapping(1.0, 1.0, 0, 0));
//    //torus1 = make_shared<rt::scale>(torus1, vector3(0.2, 0.2, 0.2));
//    //torus1 = make_shared<raytracer::rotate>(torus1, 45, 0);
//    //torus1 = make_shared<raytracer::translate>(torus1, vector3(0, 0.2, 2));
//    world.add(torus1);
//
//    // Sphere
//    shared_ptr<hittable> sphere1 = make_shared<sphere>(point3(2.0, 0.4, 0.0), 0.4, uvmapper_material, uvmapping(1.0, 1.0, 0, 0));
//    world.add(sphere1);
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.0, color(4, 4, 4), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 6, color(4, 4, 4), "SphereLight1", false));
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color::black();
//
//    return world;
//}
//
//scene scene_manager::all_materials_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto wood_texture = make_shared<image_texture>("../../data/textures/old-wood-cracked-knots.jpg");
//    auto material_ground = make_shared<lambertian>(wood_texture);
//    
//    auto dielectric_material = make_shared<dielectric>(1.5);
//    auto metal_material = make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);
//    auto lambertian_material = make_shared<lambertian>(color(0.1, 0.2, 0.9));
//    auto phong_material = make_shared<phong>(color(0.1, 0.8, 0.2), 0.1, 0.9, 0.0, 0.0);
//    auto orennayar_material = make_shared<oren_nayar>(color(0.8, 0.5, 0.5), 0.9, 0.5);
//
//    // Ground
//    world.add(make_shared<box>(point3(0, -0.8, 0), point3(10, 0.5, 40), material_ground));
//
//    world.add(make_shared<sphere>(point3(-2.2, 0.0, -1.0), 0.5, dielectric_material));
//    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, lambertian_material));
//    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, metal_material));
//    world.add(make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, phong_material));
//    world.add(make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, orennayar_material));
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.2, color(4, 4, 4), "QuadLight1"));
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    cam.background_color = color::black();
//    return world;
//}
//
//
//scene scene_manager::lambertian_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto wood_texture = make_shared<image_texture>("../../data/textures/old-wood-cracked-knots.jpg");
//    auto ground_material = make_shared<lambertian>(wood_texture);
//    
//    auto lambert_material1 = make_shared<lambertian>(color(1.0, 0.1, 0.1));
//    auto lambert_material2 = make_shared<lambertian>(color(0.1, 1.0, 0.1));
//    auto lambert_material3 = make_shared<lambertian>(color(0.3, 0.3, 0.3));
//    auto lambert_material4 = make_shared<lambertian>(color(0.1, 0.1, 1.0));
//    auto lambert_material5 = make_shared<lambertian>(make_shared<image_texture>("../../data/textures/earthmap.jpg"));
//
//    // Ground
//    world.add(make_shared<box>(point3(0, -0.8, 0), point3(10, 0.5, 40), ground_material));
//
//    auto sphere1 = make_shared<sphere>(point3(-2.2, 0.0, -1.0), 0.5, lambert_material1);
//    auto sphere2 = make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, lambert_material2);
//    auto sphere3 = make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, lambert_material3);
//    auto sphere4 = make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, lambert_material4);
//    auto sphere5 = make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, lambert_material5);
//    
//    world.add(sphere1);
//    world.add(sphere2);
//    world.add(sphere3);
//    world.add(sphere4);
//    world.add(sphere5);
//
//    // Debug
//    world.add(aabb_debug::aabb_to_box_primitive(sphere1->bounding_box()));
//    world.add(aabb_debug::aabb_to_box_primitive(sphere4->bounding_box()));
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 3, color(4, 4, 4), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 3, color(4, 4, 4), "SphereLight1"));
//
//    cam.background_color = color::black();
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::phong_spheres(perspective_camera& cam)
//{
//    scene world;
//    
//    auto diffuse_texture = make_shared<solid_color_texture>(color(0.8, 0.9, 0.3));
//    auto specular_texture = make_shared<solid_color_texture>(color(1.0, 1.0, 1.0));
//
//    auto earth_texture = make_shared<image_texture>("../../data/textures/earthmap.jpg");
//    
//    auto rocky_diffuse_texture = make_shared<image_texture>("../../data/models/rocky_diffuse.jpg");
//    auto rocky_specular_texture = make_shared<image_texture>("../../data/models/rocky_specular.jpg");
//    auto rocky_normal_texture = make_shared<normal_texture>(make_shared<image_texture>("../../data/models/normal2.jpg"), 5.0);
//    
//
//    auto wood_texture = make_shared<image_texture>("../../data/textures/old-wood-cracked-knots.jpg");
//    auto ground_material = make_shared<lambertian>(wood_texture);
//
//
//
//    auto phong_material1 = make_shared<phong>(diffuse_texture, specular_texture, color(0.0, 0.0, 0.0), 1.0);
//    auto phong_material2 = make_shared<phong>(diffuse_texture, specular_texture, color(0.0, 0.0, 0.0), 5.0);
//    auto phong_material3 = make_shared<phong>(diffuse_texture, specular_texture, color(0.0, 0.0, 0.0), 10.0);
//    auto phong_material4 = make_shared<phong>(diffuse_texture, diffuse_texture, rocky_normal_texture, color(0.0, 0.0, 0.0), 25.0);
//    auto phong_material5 = make_shared<phong>(earth_texture, specular_texture, color(0.0, 0.0, 0.0), 100.0);
//
//
//    // Ground
//    world.add(make_shared<box>(point3(0, -0.8, 0), point3(10, 0.5, 40), ground_material));
//    //world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground_material));
//
//    world.add(make_shared<sphere>(point3(-2.2, 0.0, -1.0), 0.5, phong_material1));
//    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, phong_material2));
//    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, phong_material3));
//    //world.add(make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, phong_material4));
//
//    // Load mesh
//	auto smooth_sphere = rtw_stb_obj_loader::load_model_from_file("../../data/models/smooth_sphere.obj", phong_material4, false, true);
//    smooth_sphere = make_shared<rt::scale>(smooth_sphere, vector3(0.5, 0.5, 0.5));
//    smooth_sphere = make_shared<rt::translate>(smooth_sphere, vector3(1.1, 0.0, -1.0));
//	world.add(smooth_sphere);
//
//    world.add(make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, phong_material5));
//
//    // Light Sources
//    
//    world.add(make_shared<directional_light>(point3(150, 200, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.0, color(3,3,3), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 3, color(4, 4, 4), "SphereLight1"));
//
//    cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::oren_nayar_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto ground_material = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//
//    auto oren_nayar_material1 = make_shared<oren_nayar>(color(0.4, 0.2, 1.0), 0.1, 0.0);
//    auto oren_nayar_material2 = make_shared<oren_nayar>(color(0.4, 0.2, 1.0), 0.5, 0.5);
//    auto oren_nayar_material3 = make_shared<oren_nayar>(color(0.4, 0.2, 1.0), 0.9, 1.0);
//    auto oren_nayar_material4 = make_shared<oren_nayar>(make_shared<image_texture>("../../data/textures/earthmap.jpg"), 0.9, 1.0);
//
//
//
//    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground_material));
//
//    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, oren_nayar_material1));
//    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, oren_nayar_material2));
//    world.add(make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, oren_nayar_material3));
//    world.add(make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, oren_nayar_material4));
//
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 2.2, color(0.9, 0.9, 0.9), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 8, color(4, 4, 4), "SphereLight1", false));
//
//
//
//    //cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//
//scene scene_manager::isotropic_anisotropic_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    auto ground_material = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//
//    auto isotropic_material1 = make_shared<isotropic>(color(0.4, 0.2, 1.0));
//    auto isotropic_material2 = make_shared<isotropic>(color(0.1, 0.2, 0.9));
//    auto isotropic_material3 = make_shared<isotropic>(make_shared<image_texture>("../../data/textures/shiny-aluminium.jpg"));
//    
//    auto anisotropic_material1 = make_shared<anisotropic>(color(0.4, 0.2, 1.0), 2.0);
//    auto anisotropic_material2 = make_shared<anisotropic>(color(0.1, 0.2, 0.9), 10.0);
//    auto anisotropic_material3 = make_shared<anisotropic>(make_shared<image_texture>("../../data/textures/shiny-aluminium.jpg"), 5.0);
//
//
//    world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground_material));
//
//    world.add(make_shared<sphere>(point3(-2.2, 0.0, -1.0), 0.5, isotropic_material1));
//    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, isotropic_material2));
//    world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, isotropic_material3));
//    world.add(make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, anisotropic_material1));
//    world.add(make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, anisotropic_material2));
//    world.add(make_shared<sphere>(point3(3.3, 0.0, -1.0), 0.5, anisotropic_material3));
//
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(113, 554, 0), vector3(330, 0, 0), vector3(0, 0, 305), 1, color(2, 2, 2), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 8, color(4, 4, 4), "SphereLight1", false));
//
//
//
//    cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 22;
//    cam.lookfrom = point3(0.5, 2, 9);
//    cam.lookat = point3(0.5, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//scene scene_manager::transparency_materials_spheres(perspective_camera& cam)
//{
//    scene world;
//
//    //auto ground_material = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//
//    auto wood_texture = make_shared<image_texture>("../../data/textures/uv_mapper.jpg");
//    auto ground_material = make_shared<lambertian>(wood_texture);
//
//
//
//    auto lambertian_material1 = make_shared<lambertian>(color(1.0, 0.1, 0.1), 0.5, 0.5);
//    auto lambertian_material2 = make_shared<lambertian>(make_shared<solid_color_texture>(color(0.1, 1.0, 0.1)), 0.1, 1.0);
//    //auto phong_material3 = make_shared<phong>(color(0.1, 0.1, 1.0), 0.1, 0.9, 0.0, 1.0, 0.5, 0.5);
//    //auto phong_material4 = make_shared<phong>(color(0.1, 0.1, 1.0), 0.1, 0.9, 0.0, 1.0, 0.5, 1.0);
//    //auto phong_material5 = make_shared<phong>(make_shared<image_texture>("../../data/textures/earthmap.jpg"), 0.1, 0.5, 0.025, 0.5, 0.5, 1.0);
//
//    // Ground
//    world.add(make_shared<box>(point3(0, -0.8, 0), point3(10, 0.5, 40), ground_material));
//    //world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground_material));
//
//    world.add(make_shared<sphere>(point3(-2.2, 0.0, -1.0), 0.5, lambertian_material1));
//    world.add(make_shared<sphere>(point3(-1.1, 0.0, -1.0), 0.5, lambertian_material2));
//    //world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, phong_material3));
//    //world.add(make_shared<sphere>(point3(1.1, 0.0, -1.0), 0.5, phong_material4));
//    //world.add(make_shared<sphere>(point3(2.2, 0.0, -1.0), 0.5, phong_material5));
//
//    // Light Sources
//
//    world.add(make_shared<directional_light>(point3(113, 554, 127), vector3(330, 0, 0), vector3(0, 0, 305), 1.3, color(4, 4, 4), "QuadLight1"));
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 4.0), 0.2, 3, color(4, 4, 4), "SphereLight1"));
//
//    cam.background_color = color(0, 0, 0);
//
//    cam.vfov = 18;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0.6, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//
//scene scene_manager::simple_sphere(perspective_camera& cam)
//{
//    scene world;
//
//    auto solid_material = make_shared<lambertian>(make_shared<solid_color_texture>(color(0.5, 0.5, 0.7)));
//
//    world.add(make_shared<sphere>(point3(0, 0, -1), 0.5, solid_material));
//    world.add(make_shared<sphere>(point3(0, -100.5, -1), 100, solid_material));
//
//
//    // Light Sources
//    //world.add(make_shared<omni_light>(point3(0.0, 2.0, 0.0), 0.2, 10, color(4, 4, 4), "SphereLight1", false));
//    world.add(make_shared<directional_light>(point3(113, 554, -127), vector3(330, 0, 0), vector3(0, 0, 305), 1.0, color(4, 4, 4), "QuadLight1", false));
//
//
//
//    cam.vfov = 14;
//    cam.lookfrom = point3(0, 2, 9);
//    cam.lookat = point3(0, 0, 0);
//    cam.vup = vector3(0, 1, 0);
//
//    cam.defocus_angle = 0;
//
//    return world;
//}
//
//
//
//
//
//
//
//
//
//
//scene scene_manager::alpha_texture_demo(perspective_camera& cam)
//{
//    scene world;
//
//    int width, height, bpp;
//
//
//    //const string bump_text_location = "../../data/textures/Bark_007_Height.jpg";
//    //unsigned char* bump_texture_data = stbi_load(bump_text_location.c_str(), &nxb, &nyb, &nnb, 0);
//    //if (bump_texture_data == nullptr)
//    //{
//    //    return world;
//    //}
//
//    const string alpha_text_location = "../../data/textures/alpha.png";
//    unsigned char* alpha_texture_data = stbi_load(alpha_text_location.c_str(), &width, &height, &bpp, 4);
//    if (alpha_texture_data == nullptr)
//    {
//        return world;
//    }
//
//    auto ground_material = make_shared<lambertian>(color(0.48, 0.83, 0.53));
//    //auto bark_material = make_shared<lambertian>(make_shared<image_texture>("../../data/textures/Bark_007_BaseColor_Fake.jpg"));
//    //auto solid_material = make_shared<lambertian>(make_shared<solid_color_texture>(color(0.8, 0.1, 0.1)));
//
//	auto my_alpha_texture = make_shared<alpha_texture>(alpha_texture_data, width, height, bpp);
//    auto my_alpha_material = make_shared<lambertian>(my_alpha_texture);
//
//    //auto my_bump_texture = make_shared<bump_texture>(bump_texture_data, nxb, nyb, nnb, 1.0, 20, 20);
//
//    auto wood_texture = make_shared<image_texture>("../../data/textures/old-wood-cracked-knots.jpg");
//    auto material_ground = make_shared<lambertian>(wood_texture);
//    world.add(make_shared<quad>(point3(-6, -0.5, 0), vector3(12, 0, 0), vector3(0, 0, -12), material_ground));
//
//
//	//world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, ground_material));
//	world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.74, my_alpha_material));
//
//
//    // Light Sources
//    world.add(make_shared<directional_light>(point3(343, 554, 332), vector3(-130, 0, 0), vector3(0, 0, -105), 8, color(4, 4, 4), "QuadLight1"));
//
//	cam.vfov = 12;
//	cam.lookfrom = point3(0, 2, 9);
//	cam.lookat = point3(0, 0, 0);
//	cam.vup = vector3(0, 1, 0);
//
//    cam.background_color = color(0, 0, 0);
//
//	cam.defocus_angle = 0;
//
//	return world;
//}
//
//
//scene scene_manager::load_scene(const render_parameters& params)
//{
//    scene world;
//
//    // get data from .scene file
//    scene_loader config(params.sceneName);
//    scene_builder scene = config.loadSceneFromFile();
//    imageConfig imageCfg = scene.getImageConfig();
//    cameraConfig cameraCfg = scene.getCameraConfig();
//    world.set(scene.getSceneObjects());
//
//    camera* cam = nullptr;
//
//
//
//    if (!cameraCfg.isOrthographic)
//    {
//        cam = std::make_shared<perspective_camera>();
//        cam->vfov = cameraCfg.fov;
//    }
//    else
//    {
//        cam = std::make_shared<orthographic_camera>();
//        cam->ortho_height = cameraCfg.orthoHeight;
//    }
//
//
//    cam->aspect_ratio = cameraCfg.aspectRatio;
//    cam->image_width = imageCfg.width;
//    cam->samples_per_pixel = imageCfg.spp; // denoiser quality
//    cam->max_depth = imageCfg.depth; // max nbr of bounces a ray can do
//    cam->background_color = color(0.70, 0.80, 1.00);
//    cam->lookfrom = cameraCfg.lookFrom;
//    cam->lookat = cameraCfg.lookAt;
//    cam->vup = cameraCfg.upAxis;
//    cam->is_orthographic = cameraCfg.isOrthographic;
//
//    
//    // Background
//    if (!imageCfg.background.filepath.empty())
//    {
//        auto background = std::make_shared<image_texture>(imageCfg.background.filepath);
//        cam->background_texture = background;
//        cam->background_iskybox = imageCfg.background.is_skybox;
//
//        if (imageCfg.background.is_skybox)
//            cam->background_pdf = std::make_shared<image_pdf>(background);
//    }
//    else
//    {
//        cam->background_color = imageCfg.background.rgb;
//    }
//
//
//
//    // command line parameters are stronger than .scene parameters
//    cam->aspect_ratio = params.ratio;
//    cam->image_width = params.width;
//    cam->samples_per_pixel = params.samplePerPixel; // antialiasing quality
//    cam->max_depth = params.recursionMaxDepth; // max nbr of bounces a ray can do
//
//
//    // Depth of field
//    cam->defocus_angle = cameraCfg.aperture;
//    cam->focus_dist = cameraCfg.focus;
//
//    world.set_camera(cam);
//
//    return world;
//}
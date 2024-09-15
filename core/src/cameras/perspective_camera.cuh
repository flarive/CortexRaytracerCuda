#pragma once

#include "camera.cuh"
#include "../misc/ray.cuh"
#include "../misc/constants.cuh"
#include "../misc/render_parameters.h"

#include "../samplers/random_sampler.cuh"

/// <summary>
/// Perpspective camera with target
/// </summary>
class perspective_camera : public camera
{
public:
    __device__ perspective_camera()
    {
    }

    /// <summary>
    /// Initialize camera with settings
    /// </summary>
    /// <param name="params"></param>
    __device__ void initialize(vector3 lookfrom, vector3 lookat, vector3 vup, int width, float ratio, float vfov, float aperture, float focus_dist, float ortho_height, float t0, float t1, int sqrt_spp) override;

    /// <summary>
    /// Get a randomly-sampled camera ray for the pixel at location i,j, originating from the camera defocus disk,
    /// and randomly sampled around the pixel location
    /// </summary>
    /// <param name="i"></param>
    /// <param name="j"></param>
    /// <returns></returns>
    __device__ const ray get_ray(float i, float j, int s_i, int s_j, sampler* aa_sampler, thrust::default_random_engine& rng) const override;
};




__device__ inline void perspective_camera::initialize(vector3 lookfrom, vector3 lookat, vector3 vup, int width, float ratio, float vfov, float aperture, float focus_dist, float ortho_height, float t0, float t1, int sqrt_spp)
{
    image_width = width;
    aspect_ratio = ratio;

    // Calculate the image height, and ensure that it's at least 1.
    image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    //if (!params.quietMode)
    //    std::clog << "Image : " << image_width << " x " << image_height << "\n";


    center = lookfrom;

    // Determine viewport dimensions.
    float theta = degrees_to_radians(vfov);
    float h = tan(theta / 2.0f);
    float viewport_height = 2.0f * h * focus_dist;
    float viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);


    time0 = t0;
    time1 = t1;

    lens_radius = aperture / 2;

    //sqrt_spp = static_cast<int>(glm::sqrt(samples_per_pixel));
    recip_sqrt_spp = 1.0f / sqrt_spp;

    // Calculate the u, v, w unit basis vectors for the camera coordinate frame.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(glm::cross(vup, w));
    v = glm::cross(w, u);

    float half_height = tan(theta / 2);
    float half_width = ratio * half_height;


    origin = lookfrom;
    lower_left_corner = origin
        - half_width * focus_dist * u
        - half_height * focus_dist * v
        - focus_dist * w;
    horizontal = 2 * half_width * focus_dist * u;
    vertical = 2 * half_height * focus_dist * v;



    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vector3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
    vector3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge



    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / vector3((float)image_width);
    pixel_delta_v = viewport_v / vector3((float)image_height);


    // Calculate the location of the upper left pixel.
    vector3 viewport_upper_left = center - (focus_dist * w) - viewport_u / vector3(2) - viewport_v / vector3(2);
    pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    float defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2.0f));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
}

__device__ inline const ray perspective_camera::get_ray(float s, float t, int s_i, int s_j, sampler* aa_sampler, thrust::default_random_engine& rng) const
{
    vector3 ray_direction = lens_radius * random_in_unit_disk(rng);
    vector3 offset = u * ray_direction.x + v * ray_direction.y;
    float time = time0 + get_real(rng) * (time1 - time0);

    return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
}

//__device__ inline const ray perspective_camera::get_ray(float s, float t, int s_i, int s_j, sampler* aa_sampler, thrust::default_random_engine& rng) const
//{
//    vector3 pixel_center = pixel00_loc + (vector3(s) * pixel_delta_u) + (vector3(t) * pixel_delta_v);
//
//    
//
//    // Apply antialiasing
//    vector3 pixel_sample{};
//
//    if (aa_sampler)
//    {
//        // using given anti aliasing sampler
//        pixel_sample = pixel_center + aa_sampler->generate_samples(s_i, s_j, rng);
//    }
//    else
//    {
//        // no anti aliasing
//        pixel_sample = pixel_center;
//    }
//
//
//    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rng);
//    auto ray_direction = pixel_sample - ray_origin;
//    auto ray_time = get_real(rng); // for motion blur
//
//    return ray(ray_origin, ray_direction, static_cast<int>(s), static_cast<int>(t), ray_time);
//}
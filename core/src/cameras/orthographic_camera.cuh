#pragma once

#include "camera.cuh"
#include "../misc/ray.cuh"
#include "../misc/constants.cuh"
#include "../misc/render_parameters.h"

#include "../samplers/random_sampler.cuh"

/// <summary>
/// Orthographic camera with target
/// </summary>
class orthographic_camera : public camera
{
public:
    __device__ orthographic_camera()
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



/// <summary>
/// Viewport Calculation:
/// Removed the field of view(vfov) and replaced it with a fixed orthographic viewport width(ortho_width).
/// The height of the viewport is calculated using the aspect ratio(viewport_height = viewport_width / aspect_ratio).
///
/// No Perspective Projection :
///
/// The focus_dist and perspective - related scaling(tan(theta / 2.0f) etc.) are removed since the rays in an orthographic camera are parallel.
///
/// Camera Rays :
///
/// The camera rays originate from the same point(no focal length or depth scaling).
/// The lower - left corner and the pixel deltas are calculated based on the fixed - size viewport, not focus distance or FOV.
///
/// No Depth of Field :
///
/// Orthographic cameras generally do not have depth of field, so the lens_radius is set to a fixed value.If you want to simulate defocus effects in an orthographic camera, you can adjust this as needed, but generally, it will be 0.
/// </summary>
__device__ inline void orthographic_camera::initialize(vector3 lookfrom, vector3 lookat, vector3 vup, int width, float ratio, float vfov, float aperture, float focus_dist, float ortho_height, float t0, float t1, int sqrt_spp)
{
    image_width = width;
    aspect_ratio = ratio;

    // Calculate the image height, ensuring it's at least 1.
    image_height = static_cast<int>(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    // Camera center is the lookfrom point.
    center = lookfrom;

    // Set orthographic viewport dimensions based on the specified ortho_width (instead of perspective-based FOV).
    //float viewport_width = ortho_width;
    //float viewport_height = viewport_width / aspect_ratio;

    // Determine viewport dimensions based on orthographic projection.
    float viewport_height = 2.0f * ortho_height;
    float viewport_width = viewport_height * (static_cast<float>(image_width) / image_height);

    // Set the time range for motion blur.
    time0 = t0;
    time1 = t1;

    // For orthographic cameras, lens_radius is typically 0 (no depth of field effect).
    lens_radius = aperture / 2;

    // Reciprocal of sqrt_spp (for anti-aliasing purposes).
    recip_sqrt_spp = 1.0f / sqrt_spp;

    // Calculate the camera basis vectors.
    w = unit_vector(lookfrom - lookat);  // Forward direction
    u = unit_vector(glm::cross(vup, w)); // Right direction
    v = glm::cross(w, u);                // Up direction

    // Set origin to the camera position.
    origin = lookfrom;

    // Lower-left corner of the viewport for the orthographic camera.
    lower_left_corner = origin
        - 0.5f * viewport_width * u
        - 0.5f * viewport_height * v;

    // Set the horizontal and vertical vectors across the viewport.
    horizontal = viewport_width * u;
    vertical = viewport_height * v;

    // Calculate the pixel delta (difference between pixels) for horizontal and vertical directions.
    pixel_delta_u = horizontal / vector3((float)image_width);
    pixel_delta_v = vertical / vector3((float)image_height);

    // Location of the first pixel (upper-left) in the orthographic camera view.
    vector3 viewport_upper_left = lower_left_corner + 0.5f * (pixel_delta_u + pixel_delta_v);
    pixel00_loc = viewport_upper_left;

    // Defocus disk basis vectors (typically unused in orthographic projection, set to 0).
    defocus_disk_u = vector3(0);
    defocus_disk_v = vector3(0);
}

/// <summary>
/// To convert the get_ray method from a perspective camera to an orthographic camera, we need to adjust how rays are generated.
/// In an orthographic camera, rays are parallel, meaning the direction remains constant for all rays.
/// The position of the ray, however, changes depending on the screen space coordinates (s, t), but there is no perspective distortion or depth scaling.
/// </summary>
__device__ inline const ray orthographic_camera::get_ray(float s, float t, int s_i, int s_j, sampler* aa_sampler, thrust::default_random_engine& rng) const
{
    // Generate a random point on the lens (for depth of field, if needed)
    vector3 ray_direction = lens_radius * random_in_unit_disk(rng);
    vector3 offset = u * ray_direction.x + v * ray_direction.y;

    // Ray time for motion blur (if applicable)
    float time = time0 + get_real(rng) * (time1 - time0);

    // Calculate the position on the viewport based on the normalized (s, t) coordinates
    vector3 ray_origin = lower_left_corner + s * horizontal + t * vertical;

    // The direction for orthographic camera rays is always constant: straight ahead (-w)
    vector3 ray_direction_parallel = -w;

    // Return the ray from the point on the viewport with an offset (for depth of field), and constant direction
    return ray(ray_origin + offset, ray_direction_parallel, time);
}
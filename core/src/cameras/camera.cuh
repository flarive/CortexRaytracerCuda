#pragma once

#include "../misc/ray.cuh"
#include "../misc/constants.cuh"
#include "../misc/scene.cuh"
#include "../misc/scatter_record.cuh"
#include "../misc/gpu_randomizer.cuh"
#include "../samplers/sampler.cuh"
#include "../pdf/hittable_pdf.cuh"
#include "../pdf/mixture_pdf.cuh"
#include "../materials/material.cuh"

class camera
{
public:

    float   aspect_ratio = 1.0;             // Ratio of image width over height
    int     image_width = 400;              // Rendered image width in pixel count
    int     samples_per_pixel = 10;         // Count of random samples for each pixel (antialiasing)
    int     max_depth = 10;                 // Maximum number of ray bounces into scene

    float  vfov = 90;                      // Vertical view angle (field of view) (90 is for wide-angle view for example)
    point3  lookfrom = point3(0, 0, -1);    // Point camera is looking from
    point3  lookat = point3(0, 0, 0);       // Point camera is looking at
    vector3 vup = vector3(0, 1, 0);            // Camera-relative "up" direction

    // Depth of field
    float  defocus_angle = 0;              // Variation angle of rays through each pixel
    float  focus_dist = 10;                // Distance from camera lookfrom point to plane of perfect focus

    color   background_color{0, 0, 0};               // Scene background color
    image_texture* background_texture = nullptr;
    pdf* background_pdf = nullptr;
    bool background_iskybox = false;

    bool is_orthographic = false;
    float ortho_height = 0.0;


    __device__ camera()
    {
    }

    __device__ virtual ~camera() = default;


    __device__ int getImageHeight() const
    {
        return image_height;
    }

    __device__ int getImageWidth() const
    {
        return image_width;
    }

    __device__ int getSqrtSpp() const
    {
        return sqrt_spp;
    }

    __device__ int getMaxDepth() const
    {
        return max_depth;
    }

    __device__ int getSamplePerPixel() const
    {
        return samples_per_pixel;
    }

    __device__ vector3 get_pixel_delta_u() const
    {
        return pixel_delta_u;
    }

    __device__ vector3 get_pixel_delta_v() const
    {
        return pixel_delta_v;
    }

    __device__ virtual void initialize(vector3 lookfrom, vector3 lookat, vector3 vup, int width, float ratio, float vfov, float aperture, float focus_dist, float ortho_height, float t0, float t1, int sqrt_spp) = 0;

    /// <summary>
    /// Fire a given ray and get the hit record (recursive)
    /// </summary>
    /// <param name="r"></param>
    /// <param name="world"></param>
    /// <returns></returns>
    __device__ virtual const ray get_ray(float i, float j, int s_i, int s_j, sampler* aa_sampler, curandState* local_rand_state) const = 0;

    /// <summary>
    /// Calculate ray color
    /// </summary>
    /// <param name="r"></param>
    /// <param name="depth"></param>
    /// <param name="_scene"></param>
    /// <param name="random"></param>
    /// <returns></returns>
    __device__ virtual color ray_color(const ray& r, int i, int j, int depth, int max_depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state);


    //vector3 origin;
    //vector3 lower_left_corner;
    //vector3 horizontal;
    //vector3 vertical;
    //vector3 u, v, w;
    //float time0, time1;
    //float lens_radius;

protected:
    int		image_height = 0;    // Rendered image height
    int			sqrt_spp = 0;        // Square root of number of samples per pixel
    float		recip_sqrt_spp = 0.0f;  // 1 / sqrt_spp

    point3      center{};          // Camera center
    point3      pixel00_loc{};     // Location of pixel 0, 0
    vector3     pixel_delta_u{};   // Offset to pixel to the right
    vector3     pixel_delta_v{};   // Offset to pixel below
    vector3     u{}, v{}, w{};     // Camera frame basis vectors
    vector3     defocus_disk_u{};  // Defocus disk horizontal radius
    vector3     defocus_disk_v{};  // Defocus disk vertical radius


    // new
    float time0 = 0.0f, time1 = 0.0f;
    float lens_radius = 0.0;

    vector3 origin{};
    vector3 lower_left_corner{};
    vector3 horizontal{};
    vector3 vertical{};



    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const;

    __device__ vector3 direction_from(const point3& light_pos, const point3& hit_point) const;

    __device__ color get_background_image_color(int x, int y, const vector3& unit_dir, image_texture* background_texture, bool background_iskybox);
};


/// <summary>
/// New iterative version (no recursion)
/// </summary>
__device__ inline color camera::ray_color(const ray& r, int i, int j, int depth, int max_depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state)
{
    color result_color = color::black();
    color current_attenuation = color(1.0f, 1.0f, 1.0f); // Initialize attenuation to full (no color loss)
    ray current_ray = r; // Start with the initial ray

    // Loop until we reach the max_depth or no more scattering occurs
    for (int current_depth = depth; current_depth > 0; --current_depth)
    {
        hit_record rec;

        // Check if the ray hits an object in the world
        if (!_world.hit(current_ray, interval(SHADOW_ACNE_FIX, FLT_MAX), rec, current_depth, max_depth, local_rand_state))
        {
            // If the ray hits nothing, return the background color
            vector3 unit_dir = unit_vector(current_ray.direction());
            if (background_texture)
            {
                result_color += current_attenuation * get_background_image_color(current_ray.x, current_ray.y, unit_dir, background_texture, background_iskybox);
            }
            else
            {
                result_color += current_attenuation * background_color;
            }
            break;
        }

        scatter_record srec;
        color emitted = rec.mat->emitted(current_ray, rec, rec.u, rec.v, rec.hit_point, local_rand_state);

        // Handle invisible primitives (like lights) if necessary
        if (emitted.a() == 0.0f)
        {
            _world.hit(current_ray, interval(rec.t + 0.001f, FLT_MAX), rec, current_depth, max_depth, local_rand_state);
        }

        // If the material doesn't scatter, accumulate the emitted light and stop
        if (!rec.mat->scatter(current_ray, _lights, rec, srec, local_rand_state))
        {
            result_color += current_attenuation * emitted;
            break;
        }

        // Update the result color with emitted light
        result_color += current_attenuation * emitted;

        // If there's no light sampling, handle the ray attenuation and depth
        if (srec.skip_pdf)
        {
            current_ray = srec.skip_pdf_ray;
            current_attenuation *= srec.attenuation;
            continue;
        }

        // PDF for light sampling
        hittable_pdf hpdf(_lights, rec.hit_point);
        mixture_pdf mpdf(&hpdf, srec.pdf_ptr);

        // Sample the new ray direction using mixture PDF
        ray scattered = ray(rec.hit_point, mpdf.generate(srec, local_rand_state), current_ray.time());
        float pdf_val = mpdf.value(scattered.direction(), local_rand_state);
        float scattering_pdf = rec.mat->scattering_pdf(current_ray, rec, scattered);

        // Update attenuation (how much light is lost at each bounce)
        current_attenuation *= srec.attenuation * scattering_pdf / pdf_val;

        // Update the current ray for the next iteration
        current_ray = scattered;

        // Handle transparency (alpha textures)
        bool double_sided = false;
        if (rec.mat->has_alpha_texture(double_sided))
        {
            color background_behind = rec.mat->get_diffuse_pixel_color(rec);
            ray ray_behind(rec.hit_point, current_ray.direction(), current_ray.x, current_ray.y, current_ray.time());

            hit_record rec_behind;
            if (_world.hit(ray_behind, interval(0.001f, INFINITY), rec_behind, current_depth, max_depth, local_rand_state))
            {
                scatter_record srec_behind;
                color background_infrontof = ray_color(ray_behind, i, j, current_depth - 1, max_depth, _world, _lights, local_rand_state);

                // Blend colors using alpha values
                if (double_sided)
                {
                    result_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
                }
                else
                {
                    if (rec.front_face)
                    {
                        result_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
                    }
                    else
                    {
                        result_color = background_infrontof;
                    }
                }
            }
            else
            {
                // No object behind, display the background image or color
                if (background_texture && double_sided)
                {
                    result_color = color::blend_colors(emitted + background_behind, get_background_image_color(current_ray.x, current_ray.y, unit_vector(current_ray.direction()), background_texture, background_iskybox), srec.alpha_value);
                }
                else
                {
                    result_color = get_background_image_color(current_ray.x, current_ray.y, unit_vector(current_ray.direction()), background_texture, background_iskybox);
                }
            }
        }
    }

    return result_color;
}





//__device__ inline color camera::ray_color(const ray& r, int i, int j, int depth, int max_depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state)
//{
//    // If we've exceeded the ray bounce limit, no more light is gathered.
//    if (depth <= 0)
//    {
//        return background_color;
//    }
//
//    hit_record rec;
//
//    vector3 unit_dir = unit_vector(r.direction());
//
//    // If the ray hits nothing, return the background color.
//    // 0.001 is to fix shadow acne interval
//    if (!_world.hit(r, interval(SHADOW_ACNE_FIX, FLT_MAX), rec, depth, max_depth, local_rand_state))
//    {
//        if (background_texture)
//        {
//            return get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
//        }
//        else
//        {
//            return background_color;
//        }
//    }
//
//    // ray hit a world object
//    scatter_record srec;
//    color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.hit_point, local_rand_state);
//
//    // hack for invisible primitives (such as lights)
//    if (color_from_emission.a() == 0.0f)
//    {
//        // rethrow a new ray
//        _world.hit(r, interval(rec.t + 0.001f, FLT_MAX), rec, depth, max_depth, local_rand_state);
//    }
//
//    if (!rec.mat->scatter(r, _lights, rec, srec, local_rand_state))
//    {
//        return color_from_emission;
//    }
//
//
//
//    if (_lights.object_count == 0)
//    {
//        // no lights
//        // no importance sampling
//        return srec.attenuation * ray_color(srec.skip_pdf_ray, i, j, depth - 1, max_depth, _world, _lights, local_rand_state);
//    }
//
//    // no importance sampling
//    if (srec.skip_pdf)
//    {
//        return srec.attenuation * ray_color(srec.skip_pdf_ray, i, j, depth - 1, max_depth, _world, _lights, local_rand_state);
//    }
//
//
//
//    hittable_pdf hpdf(_lights, rec.hit_point);
//
//
//    mixture_pdf mpdf;
//    if (background_texture && background_iskybox)
//    {
//        mixture_pdf p_objs(&hpdf, srec.pdf_ptr, 0.5f);
//        mpdf = mixture_pdf(new mixture_pdf(p_objs), background_pdf, 0.8f);
//    }
//    else
//    {
//        mpdf = mixture_pdf(&hpdf, srec.pdf_ptr);
//    }
//
//    ray scattered = ray(rec.hit_point, mpdf.generate(srec, local_rand_state), r.time());
//    float pdf_val = mpdf.value(scattered.direction(), local_rand_state);
//    float scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);
//
//    color final_color(0, 0, 0);
//
//    if (background_texture)
//    {
//        // with background image
//        bool double_sided = false;
//        if (rec.mat->has_alpha_texture(double_sided))
//        {
//            // render transparent object (having an alpha texture)
//            color background_behind = rec.mat->get_diffuse_pixel_color(rec);
//
//            ray ray_behind(rec.hit_point, r.direction(), r.x, r.y, r.time());
//            color background_infrontof = ray_color(ray_behind, i, j, depth - 1, max_depth, _world, _lights, local_rand_state);
//
//            hit_record rec_behind;
//            if (_world.hit(ray_behind, interval(0.001f, INFINITY), rec_behind, depth, max_depth, local_rand_state))
//            {
//                // another object is behind the alpha textured object, display it behind
//                scatter_record srec_behind;
//
//                if (double_sided)
//                {
//                    if (rec_behind.mat->scatter(ray_behind, _lights, rec_behind, srec_behind, local_rand_state))
//                    {
//                        final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
//                    }
//                }
//                else
//                {
//                    if (rec_behind.mat->scatter(ray_behind, _lights, rec_behind, srec_behind, local_rand_state) && rec.front_face)
//                    {
//                        final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
//                    }
//                    else
//                    {
//                        final_color = background_infrontof;
//                    }
//                }
//            }
//            else
//            {
//                // no other object behind the alpha textured object, just display background image
//                if (double_sided)
//                {
//                    final_color = color::blend_colors(color_from_emission + background_behind, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), i, j, depth - 1, max_depth, _world, _lights, local_rand_state), srec.alpha_value);
//                }
//                else
//                {
//                    final_color = get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
//                }
//            }
//        }
//        else
//        {
//            // render opaque object
//            color color_from_scatter = ray_color(scattered, i, j, depth - 1, max_depth, _world, _lights, local_rand_state) / pdf_val;
//            final_color = color_from_emission + srec.attenuation * scattering_pdf * color_from_scatter;
//        }
//    }
//    else
//    {
//        // with background color
//        color sample_color = ray_color(scattered, i, j, depth - 1, max_depth, _world, _lights, local_rand_state);
//        color color_from_scatter = (srec.attenuation * scattering_pdf * sample_color) / pdf_val;
//
//        bool double_sided = false;
//        if (rec.mat->has_alpha_texture(double_sided))
//        {
//            // render transparent object (having an alpha texture)
//            final_color = color::blend_colors(color_from_emission + color_from_scatter, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), i, j, depth - 1, max_depth, _world, _lights, local_rand_state), srec.alpha_value);
//        }
//        else
//        {
//            // render opaque object
//            final_color = color_from_emission + color_from_scatter;
//        }
//    }
//
//    return final_color;
//}



// old partial not recursive
//__device__ inline color camera::ray_color(const ray& r, int i, int j, int depth, int max_depth, hittable_list& _world, hittable_list& _lights, curandState* local_rand_state)
//{
//    color result_color = color::black();
//    ray current_ray = r;
//    color current_attenuation = color(1.0f, 1.0f, 1.0f);
//
//    for (int current_depth = max_depth; current_depth > 0; --current_depth)
//    {
//        hit_record rec;
//        if (!_world.hit(current_ray, interval(SHADOW_ACNE_FIX, FLT_MAX), rec, current_depth, max_depth, local_rand_state))
//        {
//            break;
//        }
//
//        scatter_record srec;
//        color emitted = rec.mat->emitted(current_ray, rec, rec.u, rec.v, rec.hit_point, local_rand_state);
//
//        if (!rec.mat->scatter(current_ray, _lights, rec, srec, local_rand_state))
//        {
//            result_color += current_attenuation * emitted;
//            break;
//        }
//
//        hittable_pdf hpdf(_lights, rec.hit_point);
//        mixture_pdf mpdf(&hpdf, srec.pdf_ptr);
//
//        ray scattered = ray(rec.hit_point, mpdf.generate(srec, local_rand_state), current_ray.time());
//        float pdf_val = mpdf.value(scattered.direction(), local_rand_state);
//        float scattering_pdf = rec.mat->scattering_pdf(current_ray, rec, scattered);
//
//        result_color += current_attenuation * emitted;
//        current_attenuation *= srec.attenuation * scattering_pdf / pdf_val;
//
//        current_ray = scattered;
//    }
//
//    return result_color;
//}






__device__ inline point3 camera::defocus_disk_sample(curandState* local_rand_state) const
{
    // Returns a random point in the camera defocus disk.
    vector3 p = random_in_unit_disk(local_rand_state);
    return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
}


__device__ inline vector3 camera::direction_from(const point3& light_pos, const point3& hit_point) const
{
    // Calculate the direction from the hit point to the light source.
    return unit_vector(light_pos - hit_point);
}

__device__ inline color camera::get_background_image_color(int x, int y, const vector3& unit_dir, image_texture* background_texture, bool background_iskybox)
{
    float u, v;

    if (background_iskybox)
        get_spherical_uv(unit_dir, background_texture->getWidth(), background_texture->getHeight(), getImageWidth(), getImageHeight(), u, v);
    else
        get_screen_uv(x, y, background_texture->getWidth(), background_texture->getHeight(), getImageWidth(), getImageHeight(), u, v);

    return background_texture->value(u, v, unit_dir);
}
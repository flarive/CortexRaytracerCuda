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

    color   background_color{};               // Scene background color
    image_texture* background_texture = nullptr;
    pdf* background_pdf = nullptr;
    bool background_iskybox = false;

    bool is_orthographic = false;
    float ortho_height = 0.0;


    __device__ camera()
    {
    }

    __device__ virtual ~camera() = default;


    __device__ const int getImageHeight() const
    {
        return image_height;
    }

    __device__ const int getImageWidth() const
    {
        return image_width;
    }

    __device__ const int getSqrtSpp() const
    {
        return sqrt_spp;
    }

    __device__ const int getMaxDepth() const
    {
        return max_depth;
    }

    __device__ const int getSamplePerPixel() const
    {
        return samples_per_pixel;
    }

    __device__ const vector3 get_pixel_delta_u() const
    {
        return pixel_delta_u;
    }

    __device__ const vector3 get_pixel_delta_v() const
    {
        return pixel_delta_v;
    }

    __device__ virtual void initialize(vector3 lookfrom, vector3 lookat, vector3 vup, int width, float ratio, float vfov, float aperture, float focus_dist, float t0, float t1, int sqrt_spp) = 0;

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
    __device__ virtual color ray_color(const ray& r, int depth, scene& _scene, curandState* local_rand_state);


    //vector3 origin;
    //vector3 lower_left_corner;
    //vector3 horizontal;
    //vector3 vertical;
    //vector3 u, v, w;
    //float time0, time1;
    //float lens_radius;

protected:
    int			image_height = 0;    // Rendered image height
    int			sqrt_spp = 0;        // Square root of number of samples per pixel
    float		recip_sqrt_spp = 0.0f;  // 1 / sqrt_spp

    point3      center{};          // Camera center
    point3      pixel00_loc{};     // Location of pixel 0, 0
    vector3     pixel_delta_u{};   // Offset to pixel to the right
    vector3     pixel_delta_v{};   // Offset to pixel below
    vector3     u{}, v{}, w{};     // Camera frame basis vectors
    vector3     defocus_disk_u{};  // Defocus disk horizontal radius
    vector3     defocus_disk_v{};  // Defocus disk vertical radius

    float time0, time1;
    float lens_radius;

    vector3 origin;
    vector3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;



    __device__ point3 defocus_disk_sample(curandState* local_rand_state) const;

    __device__ vector3 direction_from(const point3& light_pos, const point3& hit_point) const;

    __device__ color get_background_image_color(int x, int y, const vector3& unit_dir, image_texture* background_texture, bool background_iskybox);
};





__device__ inline color camera::ray_color(const ray& r, int depth, scene& _scene, curandState* local_rand_state)
{
    // If we've exceeded the ray bounce limit, no more light is gathered.
    //if (depth <= 0)
    //{
    //    // return background solid color
    //    return background_color;
    //}

    //hit_record rec;

    //vector3 unit_dir = unit_vector(r.direction());

    //hittable_list www = _scene.get_world();
    //hittable_list eee = _scene.get_emissive_objects();

    //// If the ray hits nothing, return the background color.
    //// 0.001 is to fix shadow acne interval
    //if (!www.hit(r, interval(SHADOW_ACNE_FIX, INFINITY), rec, depth, local_rand_state))
    //{
    //    if (background_texture)
    //    {
    //        return get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
    //    }
    //    else
    //    {
    //        return background_color;
    //    }
    //}

    //// ray hit a world object
    //scatter_record srec;
    //color color_from_emission = rec.mat->emitted(r, rec, rec.u, rec.v, rec.hit_point, local_rand_state);

    //// hack for invisible primitives (such as lights)
    //if (color_from_emission.a() == 0.0f)
    //{
    //    // rethrow a new ray
    //    www.hit(r, interval(rec.t + 0.001f, INFINITY), rec, depth, local_rand_state);
    //}


    ////hittable_list ppppp = _scene.get_emissive_objects();
    ////printf("found %i emissive objects\n", ppppp.object_count);

    //if (!rec.mat->scatter(r, eee, rec, srec, local_rand_state))
    //{
    //    return color_from_emission;
    //}

    //if (eee.object_count == 0)
    //{
    //    // no lights
    //    // no importance sampling
    //    return srec.attenuation * ray_color(srec.skip_pdf_ray, depth - 1, _scene, local_rand_state);
    //}

    //// no importance sampling
    //if (srec.skip_pdf)
    //    return srec.attenuation * ray_color(srec.skip_pdf_ray, depth - 1, _scene, local_rand_state);

    //auto light_ptr = new hittable_pdf(eee, rec.hit_point);

    //mixture_pdf p;

    //if (background_texture && background_iskybox)
    //{
    //    mixture_pdf p_objs(light_ptr, srec.pdf_ptr, 0.5f);
    //    p = mixture_pdf(new mixture_pdf(p_objs), background_pdf, 0.8f);
    //}
    //else
    //{
    //    p = mixture_pdf(light_ptr, srec.pdf_ptr);
    //}

    //ray scattered = ray(rec.hit_point, p.generate(srec, local_rand_state), r.time());
    //float pdf_val = p.value(scattered.direction(), local_rand_state);
    //float scattering_pdf = rec.mat->scattering_pdf(r, rec, scattered);

    //color final_color;

    //if (background_texture)
    //{
    //    // with background image
    //    bool double_sided = false;
    //    if (rec.mat->has_alpha_texture(double_sided))
    //    {
    //        // render transparent object (having an alpha texture)
    //        color background_behind = rec.mat->get_diffuse_pixel_color(rec);

    //        ray ray_behind(rec.hit_point, r.direction(), r.x, r.y, r.time());
    //        color background_infrontof = ray_color(ray_behind, depth - 1, _scene, local_rand_state);

    //        hit_record rec_behind;
    //        if (_scene.get_world().hit(ray_behind, interval(0.001f, INFINITY), rec_behind, depth, local_rand_state))
    //        {
    //            // another object is behind the alpha textured object, display it behind
    //            scatter_record srec_behind;

    //            if (double_sided)
    //            {
    //                if (rec_behind.mat->scatter(ray_behind, _scene.get_emissive_objects(), rec_behind, srec_behind, local_rand_state))
    //                {
    //                    final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
    //                }
    //            }
    //            else
    //            {
    //                if (rec_behind.mat->scatter(ray_behind, _scene.get_emissive_objects(), rec_behind, srec_behind, local_rand_state) && rec.front_face)
    //                {
    //                    final_color = color::blend_colors(background_behind, background_infrontof, srec.alpha_value);
    //                }
    //                else
    //                {
    //                    final_color = background_infrontof;
    //                }
    //            }
    //        }
    //        else
    //        {
    //            // no other object behind the alpha textured object, just display background image
    //            if (double_sided)
    //            {
    //                final_color = color::blend_colors(color_from_emission + background_behind, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), depth - 1, _scene, local_rand_state), srec.alpha_value);
    //            }
    //            else
    //            {
    //                final_color = get_background_image_color(r.x, r.y, unit_dir, background_texture, background_iskybox);
    //            }
    //        }
    //    }
    //    else
    //    {
    //        // render opaque object
    //        color color_from_scatter = ray_color(scattered, depth - 1, _scene, local_rand_state) / pdf_val;
    //        final_color = color_from_emission + srec.attenuation * scattering_pdf * color_from_scatter;
    //    }
    //}
    //else
    //{
    //    // with background color
    //    color sample_color = ray_color(scattered, depth - 1, _scene, local_rand_state);
    //    color color_from_scatter = (srec.attenuation * scattering_pdf * sample_color) / pdf_val;

    //    bool double_sided = false;
    //    if (rec.mat->has_alpha_texture(double_sided))
    //    {
    //        // render transparent object (having an alpha texture)
    //        final_color = color::blend_colors(color_from_emission + color_from_scatter, ray_color(ray(rec.hit_point, r.direction(), r.x, r.y, r.time()), depth - 1, _scene, local_rand_state), srec.alpha_value);
    //    }
    //    else
    //    {
    //        // render opaque object
    //        final_color = color_from_emission + color_from_scatter;
    //    }
    //}

    ////printf("ray_color returns %f %f %f\n", final_color.r(), final_color.g(), final_color.b());

    //return final_color;

    return color(0, 0, 0);
}



__device__ inline point3 camera::defocus_disk_sample(curandState* local_rand_state) const
{
    // Returns a random point in the camera defocus disk.
    auto p = random_in_unit_disk(local_rand_state);
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
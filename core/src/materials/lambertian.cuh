#pragma once

#include "material.cuh"
#include "../primitives/hittable.cuh"
#include "../textures/texture.cuh"

#include "../pdf/cosine_pdf.cuh"
#include "../textures/solid_color_texture.cuh"


/// <summary>
/// Diffuse material
/// More accurate representation of real diffuse objects
/// Ray is randomly scattered using Lambertian distribution
/// https://en.wikipedia.org/wiki/Lambertian_reflectance
/// The smooth diffuse material (also referred to as 'Lambertian')
/// represents an ideally diffuse material with a user - specified amount of
/// reflectance.Any received illumination is scattered so that the surface
/// looks the same independently of the direction of observation.
/// https://www.hackification.io/blog/2008/07/18/experiments-in-ray-tracing-part-4-lighting/
/// </summary>
class lambertian : public material
{
public:
    //__device__ lambertian(texture* a) : albedo(a) {}

    //__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const
    //{
    //    vector3 target = rec.hit_point + rec.normal + random_in_unit_sphere(local_rand_state);
    //    scattered = ray(rec.hit_point, target - rec.hit_point, r_in.time());
    //    attenuation = albedo->value(rec.u, rec.v, rec.hit_point);
    //    return true;
    //}

    //texture* albedo;

    __host__ __device__ lambertian(const color& _color)
        : material(new solid_color_texture(_color))
    {
    }

    __host__ __device__ lambertian(const color& _color, float _transparency, float _refraction_index)
        : material(new solid_color_texture(_color), nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, _transparency, _refraction_index)
    {
    }

    __host__ __device__ lambertian(texture* _albedo)
        : material(_albedo)
    {
    }

    __host__ __device__ lambertian(texture* _albedo, float _transparency, float _refraction_index)
        : material(_albedo, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, _transparency, _refraction_index)
    {
    }

    __host__ __device__ ~lambertian() = default;


    /// <summary>
    /// Tells how ray should be reflected when hitting a lambertian diffuse object
    /// </summary>
    /// <param name="r_in"></param>
    /// <param name="rec"></param>
    /// <param name="attenuation"></param>
    /// <param name="scattered"></param>
    /// <returns></returns>
    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const override;

    __host__ __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override;
};


__device__ inline bool lambertian::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const
{
    // Check if the material is transparent (e.g., glass)
    //if (m_transparency > 0)
    //{
    //    // Compute the refracted ray direction
    //    vector3 refracted_direction = glm::refract(r_in.direction(), rec.normal, m_refractiveIndex);
    //    srec.attenuation = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point) * color(m_transparency);
    //    srec.skip_pdf = true;
    //    srec.skip_pdf_ray = ray(rec.hit_point, refracted_direction, r_in.time());
    //    return true;

    //    // Total internal reflection (TIR)
    //    // Handle this case if needed
    //}


    srec.attenuation = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point);
    srec.pdf_ptr = new cosine_pdf(rec.normal);

    if (srec.pdf_ptr == nullptr)
        printf("-----------------------NULL cosine_pdf !!!\n");

    srec.skip_pdf = false;

    return true;




    //vector3 target = rec.hit_point + rec.normal + random_in_unit_sphere(local_rand_state);
    //srec.skip_pdf_ray = ray(rec.hit_point, target - rec.hit_point, r_in.time());
    //srec.attenuation = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point);

    ////auto zzzz = cosine_pdf(rec.normal);
    ////
    ////srec.pdf_ptr = &zzzz;
    //srec.skip_pdf = true;

    //return true;
}

__host__ __device__ inline float lambertian::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
    float cos_theta = glm::dot(rec.normal, unit_vector(scattered.direction()));
    return cos_theta < 0 ? 0.0f : cos_theta / M_PI;
}
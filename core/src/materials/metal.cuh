#pragma once

#include "material.cuh"
#include "../misc/ray.cuh"
#include "../misc/gpu_randomizer.cuh"

//class metal : public material
//{
//public:
//    __device__ metal(const vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
//    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const {
//        vector3 reflected = reflect(unitv(r_in.direction()), rec.normal);
//        scattered = ray(rec.hit_point, reflected + fuzz * random_in_unit_sphere(local_rand_state));
//        attenuation = albedo;
//        return (dot(scattered.direction(), rec.normal) > 0);
//    }
//
//    vector3 albedo;
//    float fuzz;
//};


/// <summary>
/// Metal material
/// For polished metals the ray won’t be randomly scattered
/// Ray is reflected 90°
/// color albedo -> reflective power of a surface (snow or mirror = 1, black object = 0)
/// </summary>
class metal : public material
{
public:
    __host__ __device__ metal(const color& _color, float _fuzz);

    /// <summary>
    /// Tells how ray should be reflected when hitting a metal object
    /// </summary>
    /// <param name="r_in"></param>
    /// <param name="rec"></param>
    /// <param name="attenuation"></param>
    /// <param name="scattered"></param>
    /// <returns></returns>
    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const override;

    __host__ __device__ MaterialTypeID getTypeID() const override { return MaterialTypeID::materialMetalType; }


private:
    float m_fuzz = 0.0f; // kind of blur amount (0 = none)
};


__host__ __device__ metal::metal(const color& _color, float _fuzz) : material(new solid_color_texture(_color)), m_fuzz(_fuzz < 1 ? _fuzz : 1)
{

}

__device__ bool metal::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const
{
    srec.attenuation = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point);
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;
    vector3 reflected = glm::reflect(unit_vector(r_in.direction()), rec.normal);
    srec.skip_pdf_ray = ray(rec.hit_point, reflected + m_fuzz * random_in_unit_sphere(local_rand_state), r_in.time());
    return true;
}
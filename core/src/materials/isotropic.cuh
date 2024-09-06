#pragma once

#include "../pdf/sphere_pdf.cuh"

//class isotropic : public material
//{
//public:
//    __device__ isotropic(texture* a) : albedo(a) {}
//
//    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState* local_rand_state) const {
//        scattered = ray(rec.hit_point, random_in_unit_sphere(local_rand_state), r_in.time());
//        attenuation = albedo->value(rec.u, rec.v, rec.hit_point);
//        return true;
//    }
//
//    texture* albedo;
//};

/// <summary>
/// Isotropic material
/// Isotropic materials show the same properties in all directions.
/// Glass, crystals with cubic symmetry, diamonds, metals are examples of isotropic materials.
/// </summary>
class isotropic : public material
{
public:
    __host__ __device__ isotropic(color _color);
    __host__ __device__ isotropic(texture* _albedo);

    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const override;
    __host__ __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override;

    __host__ __device__ virtual MaterialTypeID getTypeID() const { return MaterialTypeID::materialIsotropicType; }
};

__host__ __device__ isotropic::isotropic(color _color) : material(new solid_color_texture(_color))
{
}

__host__ __device__ isotropic::isotropic(texture* _albedo) : material(_albedo)
{
}

__device__ bool isotropic::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const
{
    srec.attenuation = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point);
    srec.pdf_ptr = new sphere_pdf();
    srec.skip_pdf = false;

    return true;
}

__host__ __device__ float isotropic::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
    return 1.0f / (4.0f * M_PI);
}
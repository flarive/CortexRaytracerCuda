#pragma once

#include "material.cuh"
#include "../misc/gpu_randomizer.cuh"


/// <summary>
/// Dielectric material
/// For water, glass, diamond...
/// Ray is reflected and also refracted
/// </summary>
class dielectric : public material
{
public:
    __host__ __device__ dielectric(float index_of_refraction) : ir(index_of_refraction)
    {
    }

    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const override;

    __host__ __device__ MaterialTypeID getTypeID() const override { return MaterialTypeID::materialDielectricType; }


private:
    float ir = 0.0f; // Index of Refraction (typically air = 1.0, glass = 1.3–1.7, diamond = 2.4)

    // Static methods gets constructed only once no matter how many times the function is called.
    __host__ __device__ static float reflectance(float cosine, float ref_idx);
};


__device__ bool dielectric::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const
{
    srec.attenuation = color(1.0, 1.0, 1.0);
    srec.pdf_ptr = nullptr;
    srec.skip_pdf = true;
    float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    vector3 unit_direction = unit_vector(r_in.direction());
    float cos_theta = ffmin(dot(-unit_direction, rec.normal), 1.0);
    float sin_theta = glm::sqrt(1.0 - cos_theta * cos_theta);

    bool cannot_refract = refraction_ratio * sin_theta > 1.0;
    vector3 direction;

    if (cannot_refract || reflectance(cos_theta, refraction_ratio) > get_real(local_rand_state))
        direction = glm::reflect(unit_direction, rec.normal);
    else
        direction = glm::refract(unit_direction, rec.normal, refraction_ratio);

    srec.skip_pdf_ray = ray(rec.hit_point, direction, r_in.time());
    return true;
}


// Static methods gets constructed only once no matter how many times the function is called.
__host__ __device__ float dielectric::reflectance(float cosine, float ref_idx)
{
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * (float)(pow((1 - cosine), 5));
}



//__device__ float schlick(float cosine, float ref_idx) {
//    float r0 = (1 - ref_idx) / (1 + ref_idx);
//    r0 = r0 * r0;
//    return r0 + (1 - r0) * pow((1 - cosine), 5);
//}
//
//class dielectric : public material
//{
//public:
//    __device__ dielectric(float ri) : ref_idx(ri) {}
//
//
//    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *local_rand_state) const
//    {
//        vector3 outward_normal;
//        vector3 reflected = reflect(r_in.direction(), rec.normal);
//        float ni_over_nt;
//        attenuation = vector3(1.0, 1.0, 1.0);
//        vector3 refracted;
//
//        float reflect_prob;
//        float cosine;
//
//        if (dot(r_in.direction(), rec.normal) > 0) {
//                outward_normal = -rec.normal;
//                ni_over_nt = ref_idx;
//                cosine = ref_idx * dot(r_in.direction(), rec.normal) / l2(r_in.direction());
//        }
//        else {
//                outward_normal = rec.normal;
//                ni_over_nt = 1.0 / ref_idx;
//                cosine = -dot(r_in.direction(), rec.normal) / l2(r_in.direction());
//        }
//
//        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
//            reflect_prob = schlick(cosine, ref_idx);
//        }
//        else {
//            reflect_prob = 1.0;
//        }
//
//        if (curand_uniform(local_rand_state) < reflect_prob) {
//            scattered = ray(rec.hit_point, reflected);
//        }
//        else {
//            scattered = ray(rec.hit_point, refracted);
//        }
//
//        return true;
//    }
//private:
//    float ref_idx;
//};
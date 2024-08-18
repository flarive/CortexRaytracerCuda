#pragma once

#include "material.cuh"
#include "../misc/gpu_randomizer.cuh"

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

class dielectric : public material
{
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}


    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
                            vector3& attenuation, ray& scattered, curandState *local_rand_state) const {
        vector3 outward_normal;
        vector3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vector3(1.0, 1.0, 1.0);
        vector3 refracted;

        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0) {
                outward_normal = -rec.normal;
                ni_over_nt = ref_idx;
                cosine = ref_idx * dot(r_in.direction(), rec.normal) / l2(r_in.direction());
        }
        else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0 / ref_idx;
                cosine = -dot(r_in.direction(), rec.normal) / l2(r_in.direction());
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_idx);
        }
        else {
            reflect_prob = 1.0;
        }

        if (curand_uniform(local_rand_state) < reflect_prob) {
            scattered = ray(rec.hit_point, reflected);
        }
        else {
            scattered = ray(rec.hit_point, refracted);
        }

        return true;
    }
private:
    float ref_idx;
};
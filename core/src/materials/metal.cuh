//#pragma once
//
//#include "material.cuh"
//#include "../misc/ray.cuh"
//#include "../misc/gpu_randomizer.cuh"
//
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
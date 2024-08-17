#pragma once

#include "material.cuh"
#include "../misc/ray.cuh"

class Metal : public Material {
public:
    __device__ Metal(const vector3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vector3& attenuation, Ray& scattered, curandState *local_rand_state) const {
        vector3 reflected = reflect(unitv(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

    vector3 albedo;
    float fuzz;
};
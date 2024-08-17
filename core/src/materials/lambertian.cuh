#pragma once

#include "material.cuh"
#include "../misc/vector3.cuh"
#include "../primitives/entity.cuh"
#include "../textures/texture.cuh"

class Lambertian : public Material {
public:
    __device__ Lambertian(Texture* a) : albedo(a) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vector3& attenuation, Ray& scattered, curandState *local_rand_state) const {
        vector3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = Ray(rec.p, target - rec.p, r_in.time());
        attenuation = albedo->value(rec.u, rec.v, rec.p);
        return true;
    }

    Texture* albedo;
};
#ifndef DIFFUSELIGHTH__
#define DIFFUSELIGHTH__

#include "material.cuh"
#include "texture.cuh"

class DiffuseLight : public Material {
public:
    __device__ DiffuseLight(Texture* tex) : emit(tex) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vector3& attenuation, Ray& scattered, curandState *local_rand_state) const {
        return false;
    }
    __device__ vector3 emitted(float u, float v, const vector3& p) const {
        return emit->value(u, v, p);
    }

    Texture* emit;
};

#endif
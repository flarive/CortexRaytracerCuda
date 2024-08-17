#pragma once

#include "../misc/aabb.cuh"
#include "../misc/ray.cuh"

class Material;

struct HitRecord {
    float t;
    vector3 p;
    vector3 normal;
    Material *mat_ptr;
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& r, const vector3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
    }
};

class Entity {
public:
    __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};
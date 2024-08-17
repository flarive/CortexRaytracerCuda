#pragma once

#include "../misc/aabb.cuh"
#include "../misc/ray.cuh"

class material;

struct hit_record
{
    float t;
    vector3 p;
    vector3 normal;
    material* mat_ptr;
    float u;
    float v;
    bool front_face;

    __device__ inline void set_face_normal(const ray& r, const vector3& outward_normal)
    {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
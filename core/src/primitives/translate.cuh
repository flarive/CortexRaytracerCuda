#pragma once

#include "hittable.cuh"

class translate : public hittable
{
public:
    __device__ translate(hittable* p, const vector3& displacement) : ptr(p), offset(displacement) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const;

    hittable* ptr;
    vector3 offset;
};

__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    ray moved_r(r.origin() - offset, r.direction(), r.time());
    if (!ptr->hit(moved_r, t_min, t_max, rec))
        return false;

    rec.p += offset;
    rec.set_face_normal(moved_r, rec.normal);

    return true;
}


__device__ bool translate::bounding_box(float t0, float t1, aabb& output_box) const {
    if (!ptr->bounding_box(t0, t1, output_box))
        return false;

    output_box = aabb(
        output_box.min() + offset,
        output_box.max() + offset);

    return true;
}
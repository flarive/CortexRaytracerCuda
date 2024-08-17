#pragma once

#include "hittable.cuh"

class flip_normals : public hittable
{
public:
    __device__ flip_normals(hittable* e) : ptr(e) {}

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const
    {
        if (!ptr->hit(r, t_min, t_max, rec))
            return false;

        rec.front_face = !rec.front_face;
        return true;
    }

    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const
    {
        return ptr->bounding_box(t0, t1, output_box);
    }

    hittable* ptr;
};
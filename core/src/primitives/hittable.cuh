#pragma once

#include "../misc/aabb.cuh"
#include "../misc/ray.cuh"
#include "../misc/hit_record.cuh"

class hittable
{
public:
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const = 0;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
};
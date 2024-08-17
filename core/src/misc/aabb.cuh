#pragma once

#include "ray.cuh"

__device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
__device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class aabb {
public:
    __device__ aabb() {}
    __device__ aabb(const vector3& a, const vector3& b) : _min(a), _max(b) { }

    __device__ vector3 min() const { return _min; }
    __device__ vector3 max() const { return _max; }

    __device__ bool hit(const ray& r, float tmin, float tmax) const {
        // Pixar magic?
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (min()[a] - r.origin()[a]) * invD;
            float t1 = (max()[a] - r.origin()[a]) * invD;
            float tt = t0;
            if (invD < 0.0f)
                t0 = t1;
                t1 = tt;
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin)
                return false;
        }
        return true;
    }

    vector3 _min;
    vector3 _max;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
    vector3 small(ffmin(box0.min().x, box1.min().x),
               ffmin(box0.min().y, box1.min().y),
               ffmin(box0.min().z, box1.min().z));
    vector3 big(ffmax(box0.max().x, box1.max().x),
             ffmax(box0.max().y, box1.max().y),
             ffmax(box0.max().z, box1.max().z));
    return aabb(small, big);
}
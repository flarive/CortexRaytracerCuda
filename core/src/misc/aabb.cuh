#pragma once

#include "ray.cuh"
#include "../utilities/interval.cuh"
#include "../utilities/cuda_utils.cuh"

//__host__ __device__ inline float ffmin(float a, float b) { return a < b ? a : b; }
//__host__ __device__ inline float ffmax(float a, float b) { return a > b ? a : b; }

class aabb
{
public:
    //__device__ aabb() {}
    //__device__ aabb(const vector3& a, const vector3& b) : _min(a), _max(b) { }

    //__device__ vector3 min() const { return _min; }
    //__device__ vector3 max() const { return _max; }

    //__device__ bool hit(const ray& r, float tmin, float tmax) const {
    //    // Pixar magic?
    //    for (int a = 0; a < 3; a++) {
    //        float invD = 1.0f / r.direction()[a];
    //        float t0 = (min()[a] - r.origin()[a]) * invD;
    //        float t1 = (max()[a] - r.origin()[a]) * invD;
    //        float tt = t0;
    //        if (invD < 0.0f)
    //            t0 = t1;
    //            t1 = tt;
    //        tmin = t0 > tmin ? t0 : tmin;
    //        tmax = t1 < tmax ? t1 : tmax;
    //        if (tmax <= tmin)
    //            return false;
    //    }
    //    return true;
    //}

    //vector3 _min;
    //vector3 _max;


    __host__ __device__ aabb()
    {
    }

    __host__ __device__ aabb(const interval& ix, const interval& iy, const interval& iz) : x(ix), y(iy), z(iz)
    {
    }

    __host__ __device__ aabb(const vector3& a, const vector3& b)
    {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.
        x = interval(fmin(a[0], b[0]), fmax(a[0], b[0]));
        y = interval(fmin(a[1], b[1]), fmax(a[1], b[1]));
        z = interval(fmin(a[2], b[2]), fmax(a[2], b[2]));
    }

    /// <summary>
    /// Bounding box of sub bounding boxes
    /// </summary>
    /// <param name="box0"></param>
    /// <param name="box1"></param>
    __host__ __device__ aabb(const aabb& box0, const aabb& box1)
    {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }




    __host__ __device__ const interval& axis(int n) const;


    /// <summary>
    /// Expanding bounding box to remove the possibility of numerical problems (for quad primitive for example)
    /// </summary>
    /// <returns></returns>
    __host__ __device__ aabb pad() const;

    __host__ __device__ vector3 min() const;
    __host__ __device__ vector3 max() const;



    __host__ __device__ bool hit(const ray& r, interval ray_t) const;



    interval x, y, z;
};


//__host__ __device__ inline aabb surrounding_box(aabb box0, aabb box1)
//{
//    //vector3 small(ffmin(box0.min().x, box1.min().x),
//    //           ffmin(box0.min().y, box1.min().y),
//    //           ffmin(box0.min().z, box1.min().z));
//    //vector3 big(ffmax(box0.max().x, box1.max().x),
//    //         ffmax(box0.max().y, box1.max().y),
//    //         ffmax(box0.max().z, box1.max().z));
//
//    vector3 small(0, 0, 0);
//    vector3 big(0, 0, 0);;
//
//
//    return aabb(small, big);
//}



__host__ __device__ inline const interval& aabb::axis(int n) const
{
    if (n == 1) return y;
    if (n == 2) return z;
    return x;
}

__host__ __device__ inline aabb aabb::pad() const
{
    // Return an AABB that has no side narrower than some delta, padding if necessary.
    float delta = 0.0001f;
    interval new_x = (x.size() >= delta) ? x : x.expand(delta);
    interval new_y = (y.size() >= delta) ? y : y.expand(delta);
    interval new_z = (z.size() >= delta) ? z : z.expand(delta);

    return aabb(new_x, new_y, new_z);
}

__host__ __device__ inline vector3 aabb::min() const
{
    return vector3(x.min, y.min, z.min);
}

__host__ __device__ inline vector3 aabb::max() const
{
    return vector3(x.max, y.max, z.max);
}

__host__ __device__ inline bool aabb::hit(const ray& r, interval ray_t) const
{
    for (int a = 0; a < 3; a++)
    {
        auto invD = 1 / r.direction()[a];
        auto orig = r.origin()[a];

        auto t0 = (axis(a).min - orig) * invD;
        auto t1 = (axis(a).max - orig) * invD;

        if (invD < 0)
        {
            //std::swap(t0, t1);
            float tmp = t0;
            t0 = t1;
            t1 = tmp;
        }

        if (t0 > ray_t.min) ray_t.min = t0;
        if (t1 < ray_t.max) ray_t.max = t1;

        if (ray_t.max <= ray_t.min)
            return false;
    }
    return true;
}

__host__ __device__ inline aabb operator+(const aabb& bbox, const vector3& offset)
{
    return aabb(bbox.x + offset.x, bbox.y + offset.y, bbox.z + offset.z);
}

__host__ __device__ inline aabb operator+(const vector3& offset, const aabb& bbox)
{
    return bbox + offset;
}

__host__ __device__ inline aabb operator*(const aabb& bbox, const vector3& offset)
{
    return aabb(bbox.x * offset.x, bbox.y * offset.y, bbox.z * offset.z);
}

__host__ __device__ inline aabb operator*(const vector3& offset, const aabb& bbox)
{
    return bbox * offset;
}

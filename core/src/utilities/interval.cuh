#pragma once

#include "../misc/constants.cuh"
#include <cmath>
#include "cuda_runtime.h"

class interval
{
public:
    float min, max;

    __host__ __device__ interval() : min(+INFINITY), max(-INFINITY)
    {
    }

    __host__ __device__ interval(float _min, float _max) : min(_min), max(_max)
    {
    }

    __host__ __device__ interval(const interval& a, const interval& b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max))
    {
    }

    __host__ __device__ bool contains(float x) const
    {
        // is value inside the interval ?
        return min <= x && x <= max;
    }

    __host__ __device__ bool surrounds(float x) const
    {
        // is value strictly inside the interval ?
        return min < x && x < max;
    }

    __host__ __device__ float clamp(float x) const
    {
        // clamp smaller or bigger value to the min/max interval values
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    __host__ __device__ float size() const
    {
        return max - min;
    }

    __host__ __device__ interval expand(float delta) const
    {
        auto padding = delta / 2;
        return interval(min - padding, max + padding);
    }

    __host__ __device__ static const interval get_empty();
    __host__ __device__ static const interval get_universe();
};

__host__ __device__ inline const interval interval::get_empty()
{
    return interval(+INFINITY, -INFINITY);
}

__host__ __device__ inline const interval interval::get_universe()
{
    return interval(-INFINITY, +INFINITY);
}



__host__ __device__ inline interval operator+(const interval& ival, float displacement)
{
    return interval(ival.min + displacement, ival.max + displacement);
}

__host__ __device__ inline interval operator+(float displacement, const interval& ival)
{
    return ival + displacement;
}

__host__ __device__ inline interval operator*(const interval& ival, float displacement)
{
    return interval(ival.min * displacement, ival.max * displacement);
}

__host__ __device__ inline interval operator*(float displacement, const interval& ival)
{
    return ival * displacement;
}
#pragma once

#include "vector3.cuh"

class ray
{
public:
    //__device__ ray() {}
    //__device__ ray(const vector3& a, const vector3 &b, float ti = 0.0) { A = a, B = b; _time = ti; }
    //__device__ vector3 origin() const { return A; }
    //__device__ vector3 direction() const { return B; }
    //__device__ vector3 point(float t) const { return A + t * B; }
    //__device__ float time() const { return _time; }

    int x = 0;
    int y = 0;

    __host__ __device__ ray()
    {
    }

    __host__ __device__ ray(const point3& origin, const vector3& direction) : orig(origin), dir(direction), tm(0)
    {
    }

    __host__ __device__ ray(const point3& origin, const vector3& direction, float time) : orig(origin), dir(direction), tm(time)
    {
    }

    __host__ __device__ ray(const point3& origin, const vector3& direction, int _x, int _y, float time) : orig(origin), dir(direction), x(_x), y(_y), tm(time)
    {
    }


    __host__ __device__ point3 origin() const;

    __host__ __device__ vector3 direction() const;

    __host__ __device__ float time() const;


    __host__ __device__ point3 at(float t) const;

    __host__ __device__ vector3 inverseDirection() const;

    //vector3 A, B;
    //float _time;


private:
    point3 orig{}; // origin of where the ray starts
    vector3 dir{}; // direction of ray
    float tm = 0.0f; // timestamp of the ray (when it was fired, usefull for motion blur calculation)
};


__host__ __device__ inline point3 ray::origin() const
{
    return orig;
}

__host__ __device__ inline vector3 ray::direction() const
{
    return dir;
}

__host__ __device__ inline float ray::time() const
{
    return tm;
}

__host__ __device__ inline point3 ray::at(float t) const
{
    return orig + t * dir;
}

__host__ __device__ inline vector3 ray::inverseDirection() const
{
    return 1.0f / dir;
}
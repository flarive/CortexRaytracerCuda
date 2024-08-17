#pragma once

#include "vector3.cuh"

class ray
{
public:
    __device__ ray() {}
    __device__ ray(const vector3& a, const vector3 &b, float ti = 0.0) { A = a, B = b; _time = ti; }
    __device__ vector3 origin() const { return A; }
    __device__ vector3 direction() const { return B; }
    __device__ vector3 point(float t) const { return A + t * B; }
    __device__ float time() const { return _time; }

    vector3 A, B;
    float _time;
};
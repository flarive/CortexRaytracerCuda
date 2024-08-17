//#ifndef vector3H__
//#define vector3H__
//
//#include <math.h>
//#include <stdlib.h>
//#include <iostream>
//
//class vector3 {
//public:
//    __host__ __device__ vector3() {}
//    __host__ __device__ vector3(float e0, float e1, float e2) { e[0] = e0, e[1] = e1, e[2] = e2; }
//    __host__ __device__ inline float x() const { return e[0]; }
//    __host__ __device__ inline float y() const { return e[1]; }
//    __host__ __device__ inline float z() const { return e[2]; }
//    __host__ __device__ inline float r() const { return e[0]; }
//    __host__ __device__ inline float g() const { return e[1]; }
//    __host__ __device__ inline float b() const { return e[2]; }
//
//    __host__ __device__ inline const vector3& operator+() const { return *this; }
//    __host__ __device__ inline vector3 operator-() const { return vector3(-e[0], -e[1], -e[2]); }
//    __host__ __device__ inline float operator[](int i) const { return e[i]; }
//    __host__ __device__ inline float& operator[](int i) { return e[i]; }
//
//    __host__ __device__ inline vector3& operator+=(const vector3 &v2);
//    __host__ __device__ inline vector3& operator-=(const vector3 &v2);
//    __host__ __device__ inline vector3& operator*=(const vector3 &v2);
//    __host__ __device__ inline vector3& operator/=(const vector3 &v2);
//    __host__ __device__ inline vector3& operator*=(const float t);
//    __host__ __device__ inline vector3& operator/=(const float t);
//
//    __host__ __device__ inline float l2() const {
//        return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
//    }
//
//    __host__ __device__ inline float l1() const {
//        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
//    }
//
//    __host__ __device__ inline void unitv();
//
//private:
//    float e[3];
//};
//
//inline std::istream& operator>>(std::istream &is, vector3 &t) {
//    is >> t[0] >> t[1] >> t[2];
//    return is;
//}
//
//inline std::ostream& operator<<(std::ostream &os, vector3 &t) {
//    os << t[0] << " " << t[1] << " " << t[2];
//    return os;
//}
//
//__host__ __device__ inline void vector3::unitv() {
//    float k = 1.0 / (*this).l2();
//    e[0] *= k; e[1] *= k; e[2] *= k;
//}
//
//__host__ __device__ inline vector3 operator+(const vector3 &v1, const vector3 &v2) {
//    return vector3(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
//}
//
//__host__ __device__ inline vector3 operator-(const vector3 &v1, const vector3 &v2) {
//    return vector3(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
//}
//
//__host__ __device__ inline vector3 operator*(const vector3 &v1, const vector3 &v2) {
//    return vector3(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
//}
//
//__host__ __device__ inline vector3 operator/(const vector3 &v1, const vector3 &v2) {
//    return vector3(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
//}
//
//__host__ __device__ inline vector3 operator*(float t, const vector3 &v) {
//    return vector3(t * v[0], t * v[1], t * v[2]);
//}
//
//__host__ __device__ inline vector3 operator/(vector3 v, float t) {
//    return vector3(v[0] / t, v[1] / t, v[2] / t);
//}
//
//__host__ __device__ inline vector3 operator*(const vector3 &v, float t) {
//    return vector3(t * v[0], t * v[1], t * v[2]);
//}
//
//__host__ __device__ inline float dot(const vector3 &v1, const vector3 &v2) {
//    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
//}
//
//__host__ __device__ inline vector3 cross(const vector3 &v1, const vector3 &v2) {
//    return vector3(
//        (v1[1] * v2[2] - v1[2] * v2[1]),
//        -(v1[0] * v2[2] - v1[2] * v2[0]),
//        (v1[0] * v2[1] - v1[1] * v2[0])
//    );
//}
//
//__host__ __device__ inline vector3& vector3::operator+=(const vector3 &v) {
//    e[0] += v[0];
//    e[1] += v[1];
//    e[2] += v[2];
//    return *this;
//}
//
//__host__ __device__ inline vector3& vector3::operator*=(const vector3 &v) {
//    e[0] *= v[0];
//    e[1] *= v[1];
//    e[2] *= v[2];
//    return *this;
//}
//
//__host__ __device__ inline vector3& vector3::operator/=(const vector3 &v) {
//    e[0] /= v[0];
//    e[1] /= v[1];
//    e[2] /= v[2];
//    return *this;
//}
//
//__host__ __device__ inline vector3& vector3::operator*=(const float t) {
//    e[0] *= t;
//    e[1] *= t;
//    e[2] *= t;
//    return *this;
//}
//
//__host__ __device__ inline vector3& vector3::operator/=(const float t) {
//    float k = 1.0/t;
//    e[0] *= k;
//    e[1] *= k;
//    e[2] *= k;
//    return *this;
//}
//
//__host__ __device__ inline vector3 unitv(vector3 v) {
//    return v / v.l2();
//}
//
//__host__ __device__ inline vector3 reflect(const vector3& v, const vector3& n) {
//    return v - 2 * dot(v, n) * n;
//}
//
//#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
//
//__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
//    vector3 p;
//    do {
//        p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
//    } while (p.l1() >= 1.0f);
//    return p;
//}
//
//__host__ __device__ bool refract(const vector3& v, const vector3& n, float ni_over_nt, vector3& refracted) {
//    vector3 uv = unitv(v);
//    float dt = dot(uv, n);
//    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1-dt*dt);
//    if (discriminant > 0) {
//        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
//        return true;
//    }
//    else
//        return false;
//}
//
//#endif


#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// Ensure GLM configuration is compatible with CUDA
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/detail/setup.hpp>
#include <glm/detail/type_vec3.hpp>


#include <Eigen/Eigen/Core>
#include <Eigen/Eigen/StdVector>
#include <Eigen/Eigen/Geometry>







using vector2 = glm::fvec2;
using vector3 = glm::fvec3;
using vector4 = glm::fvec4;

using matrix3 = glm::dmat3;
using matrix4 = glm::dmat4;

using point2 = glm::dvec2;
using point3 = glm::dvec3;

typedef Eigen::Matrix<float, 5, 1> Vector5d;





__host__ __device__ inline float vector_multiply_to_double(const vector3& v1, const vector3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline vector3 vector_modulo_operator(const vector3& v1, const vector3& v2)
{
	return vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

/// <summary>
/// The square of the length of this vector
/// </summary>
/// <param name="v"></param>
/// <returns></returns>
__host__ __device__ inline float vector_length_squared(vector3 v)
{
	return v.x * v.x + v.y * v.y + v.z * v.z;
}


__host__ __device__ inline float vector_length(vector3 v)
{
	return glm::sqrt(vector_length_squared(v));
}

__host__ __device__ inline vector3 unit_vector(vector3 v)
{
	return v / vector3(vector_length(v), vector_length(v), vector_length(v));
}



__host__ __device__ inline vector3 reflect(const vector3& v, const vector3& n) {
	return v - 2 * glm::dot(v, n) * n;
}

__host__ __device__ inline float l2(vector3 v) {
	return glm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

__host__ __device__ inline float l1(vector3 v) {
	return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

__host__ __device__ inline vector3 unitv(vector3 v) {
	return v / l2(v);
}


#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
    vector3 p;
    do {
        p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
    } while (l1(p) >= 1.0f);
    return p;
}



__device__ bool refract(const vector3& v, const vector3& n, float ni_over_nt, vector3& refracted) {
    vector3 uv = unitv(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}


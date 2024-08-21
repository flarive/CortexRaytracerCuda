#pragma once


#define CUDA_VERSION 12000

// Ensure GLM configuration is compatible with CUDA

//#define GLM_FORCE_INLINE
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/detail/setup.hpp>
#include <glm/detail/type_vec3.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

//#include <Eigen/Eigen/Core>
//#include <Eigen/Eigen/StdVector>
//#include <Eigen/Eigen/Geometry>







using vector2 = glm::fvec2;
using vector3 = glm::fvec3;
using vector4 = glm::fvec4;

using matrix3 = glm::fmat3;
using matrix4 = glm::fmat4;

using point2 = glm::fvec2;
using point3 = glm::fvec3;

//typedef Eigen::Matrix<float, 5, 1> Vector5d;





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

// Dot product function
__host__ __device__ inline float dot_vector(const vector3& v1, const vector3& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__host__ __device__ inline vector3 unitv(vector3 v) {
	return v / l2(v);
}

//
//#define RANDVEC3 vector3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))
//
//__device__ vector3 random_in_unit_sphere(curandState *local_rand_state) {
//    vector3 p;
//    do {
//        p = 2.0f * RANDVEC3 - vector3(1, 1, 1);
//    } while (l1(p) >= 1.0f);
//    return p;
//}





#pragma once

#include "../misc/vector3.cuh"
#include "../misc/constants.cuh"

//#include <thrust/random.h>
#include <glm/glm.hpp>




__device__ inline float get_real(curandState* local_rand_state)
{
    return curand_uniform(local_rand_state);
}

__device__ inline double get_real(curandState* local_rand_state, const double min, const double max)
{
    return min + ((max - min) * get_real(local_rand_state));
}

__device__ inline int get_int(curandState* local_rand_state, const int min, const int max)
{
    return static_cast<int>(get_real(
        local_rand_state, 
        static_cast<double>(min),
        static_cast<double>(max + 1)
    ));
}

__device__ inline vector3 get_vector3(curandState* local_rand_state)
{
    return vector3(get_real(local_rand_state), get_real(local_rand_state), get_real(local_rand_state));
}

__device__ inline vector3 get_vector3(curandState* local_rand_state, const double lower, const double upper)
{
    return vector3(
        get_real(local_rand_state, lower, upper),
        get_real(local_rand_state, lower, upper),
        get_real(local_rand_state, lower, upper)
    );
}


__device__ inline vector3 get_unit_vector(curandState* local_rand_state)
{
    const double a = get_real(local_rand_state, 0, get_2_pi());
    const double z = get_real(local_rand_state , -1, 1);
    const double r = glm::sqrt(1 - (z * z));

    return vector3(r * glm::cos(a), r * glm::sin(a), z);
}


__device__ inline vector3 random_in_unit_disk(curandState* local_rand_state)
{
    vector3 p;
    do {
        p = 2.0f * vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vector3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}


__device__ inline vector3 random_in_unit_sphere(curandState* local_rand_state)
{
    vector3 p;
    do {
        p = 2.0f * vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) - vector3(1, 1, 1);
    } while (l1(p) >= 1.0f);
    return p;
}

__device__ inline vector3 random_to_sphere(curandState* local_rand_state, double radius, double distance_squared)
{
    const double r1 = get_real(local_rand_state);
    const double r2 = get_real(local_rand_state);
    double z = 1 + r2 * (glm::sqrt(1 - radius * radius / distance_squared) - 1);

    double phi = M_2_PI * r1;
    double x = glm::cos(phi) * glm::sqrt(1 - z * z);
    double y = glm::sin(phi) * glm::sqrt(1 - z * z);

    return vector3(x, y, z);
}


__device__ inline vector3 random_on_hemisphere(curandState* local_rand_state, const vector3& normal)
{
    vector3 on_unit_sphere = get_unit_vector(local_rand_state);

    // In the same hemisphere as the normal
    if (glm::dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

__device__ inline bool refract(const vector3& v, const vector3& n, float ni_over_nt, vector3& refracted) {
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


__device__ inline vector3 random_cosine_direction(curandState* local_rand_state)
{
    const double r1 = get_real(local_rand_state);
    const double r2 = get_real(local_rand_state);
    const double phi = get_2_pi() * r1;
    const double z = glm::sqrt(1 - r2);
    const double r2_sqrt = glm::sqrt(r2);
    const double x = glm::cos(phi) * r2_sqrt;
    const double y = glm::sin(phi) * r2_sqrt;

    return vector3(x, y, z);
}


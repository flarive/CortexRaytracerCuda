#pragma once

#include "../misc/vector3.cuh"
#include "../misc/constants.cuh"

/// <summary>
/// Returns a random value between 0.0f and 1.0f
/// </summary>
/// <param name="rng"></param>
/// <returns></returns>
__host__ __device__ inline float get_real(thrust::default_random_engine& rng)
{
    // Define a uniform real distribution in the range [min_value, max_value]
    thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Generate and return a random float using the rng engine
    return dist(rng);
}

__host__ __device__ inline float get_real(thrust::default_random_engine& rng, const float min, const float max)
{
    // Define a uniform real distribution in the range [min_value, max_value]
    thrust::uniform_real_distribution<float> dist(min, max);

    // Generate and return a random float using the rng engine
    return dist(rng);
}

__host__ __device__ inline int get_int(thrust::default_random_engine& rng, const int min, const int max)
{
    // Define a uniform real distribution in the range [min_value, max_value]
    thrust::uniform_real_distribution<int> dist(min, max);

    // Generate and return a random float using the rng engine
    return dist(rng);
}

__host__ __device__ inline vector3 get_vector3(thrust::default_random_engine& rng)
{
    return vector3(get_real(rng), get_real(rng), get_real(rng));
}

__host__ __device__ inline vector3 get_vector3(thrust::default_random_engine& rng, const float lower, const float upper)
{
    return vector3(
        get_real(rng, lower, upper),
        get_real(rng, lower, upper),
        get_real(rng, lower, upper)
    );
}


__host__ __device__ inline vector3 get_unit_vector(thrust::default_random_engine& rng)
{
    const float a = get_real(rng, 0, 6.28318530718f); // M_DOUBLE_PI
    const float z = get_real(rng, -1, 1);
    const float r = glm::sqrt(1 - (z * z));

    return vector3(r * glm::cos(a), r * glm::sin(a), z);
}


__host__ __device__ inline vector3 random_in_unit_disk(thrust::default_random_engine& rng)
{
    vector3 p;
    do {
        p = 2.0f * vector3(get_real(rng), get_real(rng), 0) - vector3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}


__host__ __device__ inline vector3 random_in_unit_sphere(thrust::default_random_engine& rng)
{
    vector3 p;
    do {
        p = 2.0f * vector3(get_real(rng), get_real(rng), get_real(rng)) - vector3(1, 1, 1);
    } while (l1(p) >= 1.0f);
    return p;
}

__host__ __device__ inline vector3 random_to_sphere(thrust::default_random_engine& rng, float radius, float distance_squared)
{
    const float r1 = get_real(rng);
    const float r2 = get_real(rng);
    float z = 1 + r2 * (glm::sqrt(1 - radius * radius / distance_squared) - 1);

    float phi = 2 * M_PI * r1;
    float x = glm::cos(phi) * glm::sqrt(1 - z * z);
    float y = glm::sin(phi) * glm::sqrt(1 - z * z);

    return vector3(x, y, z);
}


__host__ __device__ inline vector3 random_on_hemisphere(thrust::default_random_engine& rng, const vector3& normal)
{
    vector3 on_unit_sphere = get_unit_vector(rng);

    // In the same hemisphere as the normal
    if (glm::dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

__host__ __device__ inline bool refract(const vector3& v, const vector3& n, float ni_over_nt, vector3& refracted) {
    vector3 uv = unitv(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}


__host__ __device__ inline vector3 random_cosine_direction(thrust::default_random_engine& rng)
{
    const float r1 = get_real(rng);
    const float r2 = get_real(rng);
    const float phi = 6.28318530718f * r1; //M_DOUBLE_PI
    const float z = glm::sqrt(1 - r2);
    const float r2_sqrt = glm::sqrt(r2);
    const float x = glm::cos(phi) * r2_sqrt;
    const float y = glm::sin(phi) * r2_sqrt;

    return vector3(x, y, z);
}

#pragma once

#include "../misc/vector3.cuh"
#include "../misc/constants.cuh"

//#include <thrust/random.h>
//#include <glm/glm.hpp>
//
//
//
//class randomizer
//{
//public:
//    __host__ __device__ randomizer(unsigned int seed) : m_rng(0), m_dist(0.0, 1.0)
//    {
//    }

    //__device__ randomizer(curandState* state) : m_state(state), m_rng(extract_seed(state)), m_dist(0.0, 1.0) 
    //{
    //}

    /// <summary>
    /// Returns a random value between 0.0f and 1.0f
    /// </summary>
    /// <param name="local_rand_state"></param>
    /// <returns></returns>
    __device__ inline float get_real(curandState* local_rand_state)
    {
        return curand_uniform(local_rand_state);
    }

    __device__ inline float get_real(curandState* local_rand_state, const float min, const float max)
    {
        return min + ((max - min) * get_real(local_rand_state));
    }

    __device__ inline int get_int(curandState* local_rand_state, const int min, const int max)
    {
        return static_cast<int>(get_real(
            local_rand_state,
            static_cast<float>(min),
            static_cast<float>(max + 1)
        ));
    }

    __device__ inline vector3 get_vector3(curandState* local_rand_state)
    {
        return vector3(get_real(local_rand_state), get_real(local_rand_state), get_real(local_rand_state));
    }

    __device__ inline vector3 get_vector3(curandState* local_rand_state, const float lower, const float upper)
    {
        return vector3(
            get_real(local_rand_state, lower, upper),
            get_real(local_rand_state, lower, upper),
            get_real(local_rand_state, lower, upper)
        );
    }


    __device__ inline vector3 get_unit_vector(curandState* local_rand_state)
    {
        const float a = get_real(local_rand_state, 0, M_DOUBLE_PI);
        const float z = get_real(local_rand_state, -1, 1);
        const float r = glm::sqrt(1 - (z * z));

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

    __device__ inline vector3 random_to_sphere(curandState* local_rand_state, float radius, float distance_squared)
    {
        const float r1 = get_real(local_rand_state);
        const float r2 = get_real(local_rand_state);
        float z = 1 + r2 * (glm::sqrt(1 - radius * radius / distance_squared) - 1);

        float phi = 2 * M_PI * r1;
        float x = glm::cos(phi) * glm::sqrt(1 - z * z);
        float y = glm::sin(phi) * glm::sqrt(1 - z * z);

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
        float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
        if (discriminant > 0) {
            refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
            return true;
        }
        else
            return false;
    }


    __device__ inline vector3 random_cosine_direction(curandState* local_rand_state)
    {
        const float r1 = get_real(local_rand_state);
        const float r2 = get_real(local_rand_state);
        const float phi = M_DOUBLE_PI * r1;
        const float z = glm::sqrt(1 - r2);
        const float r2_sqrt = glm::sqrt(r2);
        const float x = glm::cos(phi) * r2_sqrt;
        const float y = glm::sin(phi) * r2_sqrt;

        return vector3(x, y, z);
    }


//private:
//    //curandState* m_state = nullptr;
//
//    mutable thrust::default_random_engine m_rng;
//    mutable thrust::uniform_real_distribution<float> m_dist;

    //__device__ unsigned int extract_seed(curandState* state) const
    //{
    //    return curand(state);
    //}
//};
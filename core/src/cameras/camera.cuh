#pragma once

#include "../misc/ray.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"


class camera
{
public:
    //__device__ camera(vector3 lookfrom, vector3 lookat, vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1)
    //{
    //}


    

    __device__ camera()
    {
    }

    __device__ virtual ~camera() = default;

    __device__ virtual void initialize(vector3 lookfrom, vector3 lookat, vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) = 0;

    /// <summary>
    /// Fire a given ray and get the hit record (recursive)
    /// </summary>
    /// <param name="r"></param>
    /// <param name="world"></param>
    /// <returns></returns>
    //__host__ __device__ virtual const ray get_ray(int i, int j, int s_i, int s_j, sampler* aa_sampler, randomizer* rnd) const = 0;
    __device__ virtual const ray get_ray(float s, float t, curandState* local_rand_state) const = 0;


    vector3 origin;
    vector3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;
    vector3 u, v, w;
    float time0, time1;
    float lens_radius;
};
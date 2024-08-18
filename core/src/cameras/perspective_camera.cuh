#pragma once

#include "camera.cuh"
#include "../misc/ray.cuh"
#include "../misc/constants.cuh"

class perspective_camera : public camera
{
public:
    
    
    __device__ perspective_camera()
    {
    }

    __device__ void initialize(vector3 lookfrom, vector3 lookat, vector3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) override
    {
        time0 = t0;
        time1 = t1;
        lens_radius = aperture / 2;
        float theta = vfov * get_pi() / 180.0f;
        float half_height = tan(theta / 2);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unitv(lookfrom - lookat);
        u = unitv(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin
            - half_width * focus_dist * u
            - half_height * focus_dist * v
            - focus_dist * w;
        horizontal = 2 * half_width * focus_dist * u;
        vertical = 2 * half_height * focus_dist * v;
    }

    __device__ const ray get_ray(float s, float t, curandState *local_rand_state) const override
    {
        vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vector3 offset = u * rd.x + v * rd.y;
        float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
        return ray(origin + offset,
                    lower_left_corner + s*horizontal + t*vertical
                        - origin - offset, time);
    }
};
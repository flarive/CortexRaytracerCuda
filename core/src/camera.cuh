#ifndef CAMERAH__
#define CAMERAH__

#include "ray.cuh"
#include "constants.cuh"

__device__ vector3 random_in_unit_disk(curandState *local_rand_state) {
    vector3 p;
    do {
        p = 2.0f * vector3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vector3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

class Camera {
public:
    __device__ Camera(vector3 lookfrom, vector3 lookat, vector3 vup, float vfov, float aspect,
               float aperture, float focus_dist, float t0, float t1) {
        time0 = t0;
        time1 = t1;
        lens_radius = aperture / 2;
        float theta = vfov * M_PI / 180;
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
        horizontal = 2*half_width*focus_dist*u;
        vertical = 2*half_height*focus_dist*v;
    }
    __device__ Ray get_ray(float s, float t, curandState *local_rand_state) {
        vector3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vector3 offset = u * rd.x + v * rd.y;
        float time = time0 + curand_uniform(local_rand_state) * (time1 - time0);
        return Ray(origin + offset,
                    lower_left_corner + s*horizontal + t*vertical
                        - origin - offset, time);
    }

    vector3 origin;
    vector3 lower_left_corner;
    vector3 horizontal;
    vector3 vertical;
    vector3 u, v, w;
    float time0, time1;
    float lens_radius;
};

#endif
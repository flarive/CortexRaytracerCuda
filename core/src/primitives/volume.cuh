#pragma once

class volume : public hittable
{
public:
    __device__ volume(hittable* b, float f, texture* a, curandState* local_rand_state) : boundary(b), neg_inv_density(-1/f), rand_state(local_rand_state) {
        phase_function = new isotropic(a);
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const {
        return boundary->bounding_box(t0, t1, output_box);
    }

hittable* boundary;
material* phase_function;
curandState* rand_state;
float neg_inv_density;
};

__device__ bool volume::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record rec1, rec2;

    if (!boundary->hit(r, -FLT_MAX, FLT_MAX, rec1))
        return false;

    if (!boundary->hit(r, rec1.t + 0.0001, FLT_MAX, rec2))
        return false;

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = l2(r.direction());
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * log(curand_uniform(rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.p = r.point(rec.t);

    rec.normal = vector3(1, 0, 0);
    rec.front_face = true;
    rec.mat_ptr = phase_function;

    return true;
}
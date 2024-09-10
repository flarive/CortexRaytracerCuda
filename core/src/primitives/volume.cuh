#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"

/// <summary>
/// Constant Density Medium primitive (volume)
/// Usefull to create smoke/fog/mist...
/// </summary>
class volume : public hittable
{
public:
    __host__ __device__ volume(hittable* boundary, float density, texture* tex, const char* _name = "Volume");
    __host__ __device__ volume(hittable* boundary, float density, color c, const char* _name = "Volume");

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
    __host__ __device__ aabb bounding_box() const override;


private:
    hittable* m_boundary = nullptr;
    float m_neg_inv_density = 0.0f;
    material* m_phase_function = nullptr;
};


__host__ __device__ volume::volume(hittable* boundary, float density, texture* tex, const char* _name)
    : m_boundary(boundary), m_neg_inv_density(-1 / density), m_phase_function(new isotropic(tex))
{
    setName(m_name);
}

__host__ __device__ volume::volume(hittable* boundary, float density, color c, const char* _name)
    : m_boundary(boundary), m_neg_inv_density(-1 / density), m_phase_function(new isotropic(c))
{
    setName(_name);
}

__device__ bool volume::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    // Print occasional samples when debugging. To enable, set enableDebug true.
    //const bool enableDebug = false;
    const bool debugging = false;// enableDebug&& randomizer::random_double() < 0.00001f;

    hit_record rec1, rec2;

    if (!m_boundary->hit(r, interval::get_universe(), rec1, depth, max_depth, local_rand_state))
        return false;

    if (!m_boundary->hit(r, interval(rec1.t + 0.0001f, INFINITY), rec2, depth, max_depth, local_rand_state))
        return false;

    //if (debugging) std::clog << "\nray_tmin=" << rec1.t << ", ray_tmax=" << rec2.t << '\n';

    if (rec1.t < ray_t.min) rec1.t = ray_t.min;
    if (rec2.t > ray_t.max) rec2.t = ray_t.max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;



    auto ray_length = vector_length(r.direction());// .length(); ??????????
    auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    auto hit_distance = m_neg_inv_density * log(get_real(local_rand_state));

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.hit_point = r.at(rec.t);

    //if (debugging) {
    //    std::clog << "hit_distance = " << hit_distance << '\n'
    //        << "rec.t = " << rec.t << '\n'
    //        << "rec.hit_point = " << rec.hit_point << '\n';
    //}

    rec.normal = vector3(1, 0, 0);  // arbitrary
    rec.front_face = true;     // also arbitrary
    rec.mat = m_phase_function;
    rec.name = m_name;
    //rec.bbox = bbox;

    return true;
}

__host__ __device__ aabb volume::bounding_box() const
{
    return m_boundary->bounding_box();
}
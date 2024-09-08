#pragma once

#include "hittable.cuh"

namespace rt
{
    class flip_normals : public hittable
    {
    public:
        __host__ __device__ flip_normals(hittable* e);

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
        __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;
        __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;
        __host__ __device__ aabb bounding_box() const override;

        __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTransformFlipNormalType; }

    private:
        hittable* m_object;
    };
}



__host__ __device__ rt::flip_normals::flip_normals(hittable* p) : m_object(p)
{
    setName(p->getName());

    m_bbox = m_object->bounding_box();
}

__device__ bool rt::flip_normals::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    if (m_object->hit(r, ray_t, rec, depth, max_depth, local_rand_state))
    {
        rec.normal = -rec.normal;
        return true;
    }
    else
        return false;
}

__device__ float rt::flip_normals::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 rt::flip_normals::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3();
}

__host__ __device__ aabb rt::flip_normals::bounding_box() const
{
    return m_bbox;
}
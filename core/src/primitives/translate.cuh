#pragma once

#include "hittable.cuh"

namespace rt
{
    /// <summary>
    /// Translate an instance
    /// </summary>
    class translate : public hittable
    {
    public:
        __host__ __device__ translate(hittable* p, const vector3& displacement);

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;
        __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;
        __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;
        __host__ __device__ aabb bounding_box() const override;

        __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTransformTranslateType; }

    private:
        hittable* m_object;
        vector3 m_offset{};
    };
}



__host__ __device__ rt::translate::translate(hittable* p, const vector3& displacement)
    : m_object(p), m_offset(displacement)
{
    setName(p->getName());

    m_bbox = m_object->bounding_box() + m_offset;
}

__device__ bool rt::translate::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    // Move the ray backwards by the offset
    ray offset_r(r.origin() - m_offset, r.direction(), r.time());

    // Determine where (if any) an intersection occurs along the offset ray
    if (!m_object->hit(offset_r, ray_t, rec, depth, local_rand_state))
        return false;

    // Move the intersection point forwards by the offset
    rec.hit_point += m_offset;

    return true;
}

__device__ float  rt::translate::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 rt::translate::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3();
}

__host__ __device__ aabb rt::translate::bounding_box() const
{
    return m_bbox;
}



//
//__device__ bool translate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//    ray moved_r(r.origin() - offset, r.direction(), r.time());
//    if (!ptr->hit(moved_r, t_min, t_max, rec))
//        return false;
//
//    rec.p += offset;
//    rec.set_face_normal(moved_r, rec.normal);
//
//    return true;
//}
//
//
//__device__ bool translate::bounding_box(float t0, float t1, aabb& output_box) const {
//    if (!ptr->bounding_box(t0, t1, output_box))
//        return false;
//
//    output_box = aabb(
//        output_box.min() + offset,
//        output_box.max() + offset);
//
//    return true;
//}
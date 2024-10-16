#pragma once

#include "hittable.cuh"

namespace rt
{
    /// <summary>
    /// https://github.com/njeff/raytracer0/blob/27de69303cd25c77b126391fdd4c7c24b4ff3de7/week/Hittable.java#L67
    /// </summary>
    class scale : public hittable
    {
    public:
		__host__ __device__ scale(hittable* p, const vector3& _scale);
		__device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;
		__device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;
		__device__ vector3 random(const vector3& o, thrust::default_random_engine& rng) const override;
		__host__ __device__ aabb bounding_box() const override;

		__host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTransformScaleType; }

    private:
		hittable* m_object = nullptr;
        vector3 m_pivot{};
        vector3 m_scale{};
    };
}



__host__ __device__ inline rt::scale::scale(hittable* p, const vector3& _scale)
	: m_object(p), m_scale(_scale)
{
	m_name = p->getName();

	// Calculate new bounding box after scaling
	m_bbox = m_object->bounding_box(); // Get original bounding box

	// Apply scaling to the bounding box
	m_bbox.x.min *= m_scale.x;
	m_bbox.x.max *= m_scale.x;

	m_bbox.y.min *= m_scale.y;
	m_bbox.y.max *= m_scale.y;

	m_bbox.z.min *= m_scale.z;
	m_bbox.z.max *= m_scale.z;
}

__device__ inline bool rt::scale::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
	// Apply scaling to ray's origin and direction
	vector3 origin = r.origin() / m_scale;
	vector3 direction = r.direction() / m_scale;

	ray scaled_r = ray(origin, direction, r.time());
	if (m_object->hit(scaled_r, ray_t, rec, depth, max_depth, rng))
	{
		// Scale hit point and normal back to the original scale
		rec.hit_point *= m_scale;

		// Calculate normal in the scaled space
		vector3 normal = rec.normal / m_scale;
		rec.normal = unit_vector(normal);

		return true;
	}

	return false;
}

__device__ inline float rt::scale::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
    return 0.0f;
}

__device__ inline vector3 rt::scale::random(const vector3& o, thrust::default_random_engine& rng) const
{
	return vector3();
}

__host__ __device__ inline aabb rt::scale::bounding_box() const
{
    return m_bbox;
}
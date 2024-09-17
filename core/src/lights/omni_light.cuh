#pragma once

#include "../materials/diffuse_light.cuh"

/// <summary>
/// Omni directional light
/// </summary>
class omni_light : public light
{
public:
    __host__ __device__ omni_light(point3 _position, float _radius, float _intensity, color _color, const char* _name = "SphereLight", bool _invisible = true)
        : light(_position, _intensity, _color, _invisible, _name)
    {
        radius = _radius;

        m_mat = new diffuse_light(m_color, _intensity, false, m_invisible);

        // calculate stationary sphere bounding box for ray optimizations
        vector3 rvec = vector3(radius, radius, radius);
        m_bbox = aabb(m_position - rvec, m_position + rvec);
    }


    __host__ __device__ aabb bounding_box() const override;

    /// <summary>
    /// Logic of sphere ray hit detection
    /// </summary>
    /// <param name="r"></param>
    /// <param name="ray_t"></param>
    /// <param name="rec"></param>
    /// <returns></returns>
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;


    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;


    /// <summary>
    /// Random special implementation for sphere lights (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const point3& o, thrust::default_random_engine& rng) const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::lightOmniType; }


private:
    float radius = 0.0f;
    vector3 center_vec{};
};



__device__ inline bool omni_light::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
    point3 center = m_position;
    vector3 oc = r.origin() - center;
    auto a = vector_length_squared(r.direction());
    auto half_b = glm::dot(oc, r.direction());
    auto c = vector_length_squared(oc) - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }


    // Hide light source
    if (m_invisible && depth == max_depth)
    {
        return false;
    }


    // number of hits encountered by the ray (only the nearest ?)
    rec.t = root;

    // point coordinate of the hit
    rec.hit_point = r.at(rec.t);

    // material of the hit object
    rec.mat = m_mat;

    // name of the primitive hit by the ray
    rec.name = m_name;
    rec.bbox = m_bbox;

    // set normal and front-face tracking
    vector3 outward_normal = (rec.hit_point - center) / radius;
    rec.set_face_normal(r, outward_normal);

    // UV coordinates
    const uvmapping mapping = uvmapping();
    get_sphere_uv(outward_normal, rec.u, rec.v, mapping);

    return true;
}

__device__ inline float omni_light::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
    // This method only works for stationary spheres.
    hit_record rec;
    if (!this->hit(ray(o, v), interval(SHADOW_ACNE_FIX, INFINITY), rec, 0, max_depth, rng))
        return 0;

    auto cos_theta_max = sqrt(1 - radius * radius / vector_length_squared(m_position - o));
    auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

    return 1.0f / solid_angle;
}

__device__ inline vector3 omni_light::random(const point3& o, thrust::default_random_engine& rng) const
{
    vector3 direction = m_position - o;
    float distance_squared = vector_length_squared(direction);
    onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(rng, radius, distance_squared));
}

__host__ __device__ inline aabb omni_light::bounding_box() const
{
    return m_bbox;
}


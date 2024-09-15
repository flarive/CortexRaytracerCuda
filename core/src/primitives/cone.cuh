#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"

/// <summary>
/// Cone primitive
/// https://github.com/iceman201/RayTracing/blob/master/Ray%20tracing/Cone.cpp
/// </summary>
class cone : public hittable
{
public:
    __host__ __device__ cone(const char* _name = "Cone");
    __host__ __device__ cone(vector3 _center, float _radius, float _height, material* _material, const char* _name = "Cone");
    __host__ __device__ cone(vector3 _center, float _radius, float _height, material* _material, const uvmapping& _mapping, const char* _name = "Cone");
    virtual ~cone() {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;
    __device__ vector3 random(const vector3& o, thrust::default_random_engine& rng) const override;

    __host__ __device__ aabb bounding_box() const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableConeType; }

private:
    vector3 center{};
    float radius = 0.0f;
    float height = 0.0f;
    material* mat = nullptr;
};


__host__ __device__ cone::cone(const char* _name)
    : cone(vector3(0), 1, 1, nullptr, uvmapping(), _name)
{
};

__host__ __device__ cone::cone(vector3 _center, float _radius, float _height, material* _material, const char* _name)
    : cone(_center, _radius, _height, _material, uvmapping(), _name)
{
};

__host__ __device__ cone::cone(vector3 _center, float _radius, float _height, material* _material, const uvmapping& _mapping, const char* _name)
    : center(_center), radius(_radius), height(_height), mat(_material)
{
    m_mapping = _mapping;
    setName(_name);

    // calculate cone bounding box for ray optimizations
    m_bbox = aabb(
        vector3(center.x - radius, center.y, center.z - radius),
        vector3(center.x + radius, center.y + height, center.z + radius)
    );
}

__device__ bool cone::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
    double A = r.origin().x - center.x;
    double B = r.origin().z - center.z;
    double D = height - r.origin().y + center.y;

    double tan = (radius / height) * (radius / height);

    double a = (r.direction().x * r.direction().x) + (r.direction().z * r.direction().z) - (tan * (r.direction().y * r.direction().y));
    double b = (2 * A * r.direction().x) + (2 * B * r.direction().z) + (2 * tan * D * r.direction().y);
    double c = (A * A) + (B * B) - (tan * (D * D));

    double delta = b * b - 4 * (a * c);
    if (fabs(delta) < 0.001) return false;

    if (delta < 0)
    {
        return false;
    }

    double root = (-b - sqrt(delta)) / (2 * a);

    if (root < ray_t.min || ray_t.max < root)
    {
        root = (-b + sqrt(delta)) / (2 * a);
        if (root < ray_t.min || ray_t.max < root)
        {
            return false;
        }
    }


    double y = r.origin().y + root * r.direction()[1];

    if ((y < center.y) || (y > center.y + height))
    {
        return false;
    }


    rec.t = root;
    rec.hit_point = r.at(rec.t);


    // Calculate the outward normal
    vector3 outward_normal;
    if (rec.hit_point.y <= center.y) {
        // Point lies on the base of the cone
        outward_normal = vector3(0, -1, 0);
    }
    else if (rec.hit_point.y >= center.y + height) {
        // Point lies on the top of the cone
        outward_normal = vector3(0, 1, 0);
    }
    else {
        // Point lies on the curved surface of the cone
        double rs = sqrt((rec.hit_point.x - center.x) * (rec.hit_point.x - center.x) + (rec.hit_point.z - center.z) * (rec.hit_point.z - center.z));
        outward_normal = vector3(rec.hit_point.x - center.x, rs * (radius / height), rec.hit_point.z - center.z);
    }

    rec.set_face_normal(r, glm::normalize(outward_normal));



    get_cone_uv(outward_normal, rec.u, rec.v, radius, height, m_mapping);
    rec.mat = mat;
    rec.name = m_name;
    rec.bbox = m_bbox;

    return true;
}

__device__ float cone::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
    return 0.0f;
}

__device__ vector3 cone::random(const vector3& o, thrust::default_random_engine& rng) const
{
    return vector3(1, 0, 0);
}


__host__ __device__ aabb cone::bounding_box() const
{
    return m_bbox;
}
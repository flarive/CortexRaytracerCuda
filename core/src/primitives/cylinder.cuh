#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"

/// <summary>
/// Cylinder primitive
/// </summary>
class cylinder : public hittable
{
public:
    __host__ __device__ cylinder(const char* _name = "Cylinder");
    __host__ __device__ cylinder(point3 _center, float _radius, float _height, const char* _name = "Cylinder");
    __host__ __device__ cylinder(point3 _center, float _radius, float _height, material* _material, const char* _name = "Cylinder");
    __host__ __device__ cylinder(point3 _center, float _radius, float _height, material* _material, const uvmapping& _mapping, const char* _name = "Cylinder");

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;
    __device__ vector3 random(const vector3& o, thrust::default_random_engine& rng) const override;

    __host__ __device__ aabb bounding_box() const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableCylinderType; }


private:
    point3 center{};
    float radius = 0.0f;
    float height = 0.0f;
    material* mat = nullptr;
};



__host__ __device__ inline cylinder::cylinder(const char* _name)
    : cylinder(vector3(0, 0, 0), 0, 0, nullptr, uvmapping(), _name)
{
}

__host__ __device__ inline cylinder::cylinder(point3 _center, float _radius, float _height, const char* _name)
    : cylinder(_center, _radius, _height, nullptr, uvmapping(), _name)
{
}

__host__ __device__ inline cylinder::cylinder(point3 _center, float _radius, float _height, material* _material, const char* _name)
    : cylinder(_center, _radius, _height, _material, uvmapping(), _name)
{
}

__host__ __device__ inline cylinder::cylinder(point3 _center, float _radius, float _height, material* _material, const uvmapping& _mapping, const char* _name)
    : center(_center), radius(_radius), height(_height), mat(_material)
{
    setName(_name);
    m_mapping = _mapping;

    // calculate cylinder bounding box for ray optimizations
    m_bbox = aabb(
        point3(center.x - radius, center.y, center.z - radius),
        point3(center.x + radius, center.y + height, center.z + radius)
    );
}

__device__ inline bool cylinder::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
    vector3 oc = r.origin() - center;
    float a = r.direction().x * r.direction().x + r.direction().z * r.direction().z;
    float b = 2.0f * (oc.x * r.direction().x + oc.z * r.direction().z);
    float c = oc.x * oc.x + oc.z * oc.z - radius * radius;
    float d = b * b - 4 * a * c;

    if (d < 0)
    {
        return false;
    }

    float root = (-b - sqrt(d)) / (2.0f * a);

    if (root < ray_t.min || ray_t.max < root)
    {
        root = (-b + sqrt(d)) / (2.0f * a);
        if (root < ray_t.min || ray_t.max < root)
        {
            return false;
        }
    }

    double y = r.origin().y + root * r.direction().y;

    if ((y < center.y) || (y > center.y + height))
    {
        return false;
    }

    rec.t = root;
    rec.hit_point = r.at(rec.t);
    rec.normal = vector3((rec.hit_point.x - center.x) / radius, 0, (rec.hit_point.z - center.z) / radius);
    vector3 outward_normal = (rec.hit_point - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_cylinder_uv(outward_normal, rec.u, rec.v, radius, height, m_mapping);
    rec.mat = mat;
    rec.name = m_name;
    rec.bbox = m_bbox;

    return true;
}

__device__ inline float cylinder::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
    return 0.0f;
}

__device__ inline vector3 cylinder::random(const vector3& o, thrust::default_random_engine& rng) const
{
    return vector3(1, 0, 0);
}

__host__ __device__ inline aabb cylinder::bounding_box() const
{
    return m_bbox;
}
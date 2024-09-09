#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"


class disk : public hittable
{
public:
    __host__ __device__ disk(const char* _name = "Disk");
    __host__ __device__ disk(point3 _center, float _radius, float _height, material* _mat, const char* _name = "Disk");
    __host__ __device__ disk(point3 _center, float _radius, float _height, material* _mat, const uvmapping& _mapping, const char* _name = "Disk");

    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
    __host__ __device__ virtual aabb bounding_box() const override;

public:
    point3 center{};
    float radius = 0.0f;
    float height = 0.0f;
    material* mat = nullptr;
};



__host__ __device__ disk::disk(const char* _name)
    : disk(vector3(0, 0, 0), 1.0, 2.0, nullptr, uvmapping(), _name)
{
}

__host__ __device__ disk::disk(point3 _center, float _radius, float _height, material* _mat, const char* _name)
    : disk(_center, _radius, _height, _mat, uvmapping(), _name)
{
}

__host__ __device__ disk::disk(point3 _center, float _radius, float _height, material* _mat, const uvmapping& _mapping, const char* _name)
    : center(_center), radius(_radius), height(_height), mat(_mat)
{
    setName(_name);
    m_mapping = _mapping;

    // calculate disk bounding box for ray optimizations
    m_bbox = aabb(center - vector3(radius, height / 2, radius), center + vector3(radius, height / 2, radius));
}

__device__ bool disk::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    // Compute the intersection with the plane containing the disk
    float t = (center.y - r.origin().y) / r.direction().y;
    if (t < ray_t.min || t > ray_t.max)
        return false;

    point3 hit_point = r.at(t);

    // Check if the hit point is within the disk's radius
    double dist_squared = (hit_point.x - center.x) * (hit_point.x - center.x) + (hit_point.z - center.z) * (hit_point.z - center.z);
    if (dist_squared > radius * radius)
        return false;

    rec.t = t;
    rec.hit_point = hit_point;

    vector3 outward_normal = glm::normalize(vector3(0, 1, 0)); // Disk is in the XY plane, normal is in Y direction

    rec.set_face_normal(r, outward_normal);

    // Transform the hit point into the local disk coordinates
    vector3 local_hit_point = hit_point - center;

    get_disk_uv(local_hit_point, rec.u, rec.v, radius, m_mapping);

    rec.mat = mat;
    rec.name = m_name;
    rec.bbox = m_bbox;

    return true;
}

__host__ __device__ aabb disk::bounding_box() const
{
    return m_bbox;
}
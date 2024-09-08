#pragma once

#include "flip_normals.cuh"
#include "../misc/aabb.cuh"
#include "../utilities/cuda_utils.cuh"

class box : public hittable
{
public:
    __host__ __device__ box(const char* _name = "Box");
    __host__ __device__ box(const vector3& _center, const vector3& _size, material* _mat, const char* _name = "Box");
    __host__ __device__ box(const vector3& _center, const vector3& _size, material* _mat, const uvmapping& _mapping, const char* _name = "Box");

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const override;
    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableBoxType; }

    __host__ __device__ aabb bounding_box() const override;

private:
    vector3 pmin{}, pmax{};
    hittable_list* list_ptr;
};



__host__ __device__ box::box(const char* _name)
    : box(vector3(0, 0, 0), vector3(0, 0, 0), nullptr, uvmapping(), _name)
{
}

__host__ __device__ box::box(const vector3& _center, const vector3& _size, material* _mat, const char* _name)
    : box(_center, _size, _mat, uvmapping(), _name)
{
}

__host__ __device__ box::box(const vector3& _center, const vector3& _size, material* _mat, const uvmapping& _mapping, const char* _name)
{
    setName(_name);

    pmin = vector3(_center.x - (_size.x / 2.0), _center.y - (_size.y / 2.0), _center.z - (_size.z / 2.0));
    pmax = pmin + _size;

    list_ptr = new hittable_list();

    // font face
    list_ptr->add(new xy_rect(pmin.x, pmax.x, pmin.y, pmax.y, pmax.z, _mat, _mapping, concat(_name, "_front")));

    // back face
    list_ptr->add(new rt::flip_normals(new xy_rect(pmin.x, pmax.x, pmin.y, pmax.y, pmin.z, _mat, _mapping, concat(_name, "back"))));

    // top face
    list_ptr->add(new xz_rect(pmin.x, pmax.x, pmin.z, pmax.z, pmax.y, _mat, _mapping, concat(_name, "_face")));

    // bottom face
    list_ptr->add(new rt::flip_normals(new xz_rect(pmin.x, pmax.x, pmin.z, pmax.z, pmin.y, _mat, _mapping, concat(_name, "_bottom"))));

    // right face
    list_ptr->add(new yz_rect(pmin.y, pmax.y, pmin.z, pmax.z, pmax.x, _mat, _mapping, concat(_name, "_right")));

    // left face
    list_ptr->add(new rt::flip_normals(new yz_rect(pmin.y, pmax.y, pmin.z, pmax.z, pmin.x, _mat, _mapping, concat(_name, "_left"))));

    m_bbox = aabb(pmin, pmax);
}

__device__ bool box::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    return list_ptr->hit(r, ray_t, rec, depth, max_depth, local_rand_state);
}

__device__ float box::pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 box::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3();
}

aabb box::bounding_box() const
{
    return m_bbox;
}

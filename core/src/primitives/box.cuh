//#pragma once
//
//#include "flip_normals.cuh"
//#include "../misc/aabb.cuh"
//
//
//
//class box : public hittable
//{
//public:
//    __device__ box() {}
//    __device__ box(const vector3& p0, const vector3& p1, material* ptr);
//
//    __device__ virtual bool hit(const ray& r, float t0, float t1, hit_record& rec) const;
//
//    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const {
//        output_box = aabb(box_min, box_max);
//        return true;
//    }
//
//    vector3 box_min;
//    vector3 box_max;
//    hittable_list sides;
//};
//
//__device__ box::box(const vector3& p0, const vector3& p1, material* ptr)
//{
//    box_min = p0;
//    box_max = p1;
//
//    sides.add(new xy_rect(p0.x, p1.x, p0.y, p1.y, p1.z, ptr));
//    sides.add(new flip_normals(new xy_rect(p0.x, p1.x, p0.y, p1.y, p0.z, ptr)));
//    sides.add(new xz_rect(p0.x, p1.x, p0.z, p1.z, p1.y, ptr));
//    sides.add(new flip_normals(new xz_rect(p0.x, p1.x, p0.z, p1.z, p0.y, ptr)));
//    sides.add(new yz_rect(p0.y, p1.y, p0.z, p1.z, p1.x, ptr));
//    sides.add(new flip_normals(new yz_rect(p0.y, p1.y, p0.z, p1.z, p0.x, ptr)));
//}
//
//__device__ bool box::hit(const ray& r, float t0, float t1, hit_record& rec) const
//{
//    return sides.hit(r, t0, t1, rec);
//}
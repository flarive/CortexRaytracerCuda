#pragma once

#include "hittable.cuh"

class hittable_list: public hittable
{
public:
    __device__ hittable_list() {}
    __device__ hittable_list(hittable **e, int n) { list = e; list_size = n; allocated_list_size = n; } 
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    __device__ void add(hittable* e);

    hittable **list;
    int list_size;
    int allocated_list_size;
};

__device__ bool hittable_list::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = tmax;

    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}

__device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box) const {
    if (list_size < 1) {
        return false;
    }

    aabb temp_box;
    bool first_true = list[0]->bounding_box(t0, t1, temp_box);
    if (!first_true) {
        return false;
    } else {
        box = temp_box;
    }
    
    for (int i = 1; i < list_size; i++) {
        if (list[i]->bounding_box(t0, t1, temp_box)) {
            box = surrounding_box(box, temp_box);
        } else {
            return false;
        }
    }
    return true;
}

__device__ void hittable_list::add(hittable* e) {
    if (allocated_list_size <= list_size) {
        hittable** new_list = new hittable*[list_size*2];
        for (int i = 0; i < list_size; i++) {
            new_list[i] = list[i];
        }
        list = new_list;
        allocated_list_size = list_size * 2;
    }
    list[list_size++] = e;
}
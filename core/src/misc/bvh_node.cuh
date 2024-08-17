#pragma once

#include "ray.cuh"
#include "../primitives/hittable.cuh"

enum axis { X, Y, Z };

__device__ void swap(hittable*& p1, hittable*& p2) {
    hittable* temp = p1;
    *p1 = *p2;
    p2 = temp;
}

template<axis _axis>
__device__ void bubble_sort(hittable** e, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            aabb box_left, box_right;
            hittable *ah = e[j];
            hittable *bh = e[j+1];
            
            ah->bounding_box(0, 0, box_left);
            bh->bounding_box(0, 0, box_right);

            if ((_axis == X && (box_left.min().x - box_right.min().x) < 0.0)
             || (_axis == Y && (box_left.min().y - box_right.min().y) < 0.0)
             || (_axis == Z && (box_left.min().z - box_right.min().z) < 0.0)) {
                swap(e[j], e[j+1]);
            }
        }
    }
}

class bvh_node: public hittable {
public:
    __device__ bvh_node() {}
    __device__ bvh_node(hittable **e, int n, float time0, float time1, curandState& local_rand_state);

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;

    hittable *left;
    hittable *right;
    aabb box;
};

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
    b = box;
    return true;
}

__device__ bool bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    if (box.hit(r, tmin, tmax)) {
        hit_record left_rec, right_rec;
        bool hit_left = left->hit(r, tmin, tmax, left_rec);
        bool hit_right = right->hit(r, tmin, tmax, right_rec);
        if (hit_left && hit_right) {
            if (left_rec.t < right_rec.t) {
                rec = left_rec;
            } else {
                rec = right_rec;
            }
            return true;
        } else if (hit_left) {
            rec = left_rec;
            return true;
        } else if (hit_right) {
            rec = right_rec;
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}

__device__ bvh_node::bvh_node(hittable **e, int n, float time0, float time1, curandState& local_rand_state) {
    int axis = int(3 * curand_uniform(&local_rand_state));

    if (axis == 0)
        bubble_sort<X>(e, n);
    else if (axis == 1)
        bubble_sort<Y>(e, n);
    else
        bubble_sort<Z>(e, n);

    if (n == 1) {
        left = right = e[0];
    }
    else if (n == 2) {
        left = e[0];
        right = e[1];
    }
    else {
        left = new bvh_node(e, n/2, time0, time1, local_rand_state);
        right = new bvh_node(e + n/2, n - n/2, time0, time1, local_rand_state);
    }

    aabb box_left, box_right;

    box = surrounding_box(box_left, box_right);
}
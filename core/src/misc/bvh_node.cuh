#pragma once

#include "ray.cuh"
#include "../primitives/hittable.cuh"
#include <thrust/random.h>  // Include Thrust random library

enum axis2 { X, Y, Z };

__device__ void inline swap(hittable*& p1, hittable*& p2) {
    hittable* temp = p1;
    *p1 = *p2;
    p2 = temp;
}

template<axis2 _axis>
__device__ void bubble_sort(hittable** e, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            aabb box_left, box_right;
            hittable* ah = e[j];
            hittable* bh = e[j + 1];

            box_left = ah->bounding_box();
            box_right = bh->bounding_box();

            if ((_axis == X && (box_left.min().x - box_right.min().x) < 0.0)
                || (_axis == Y && (box_left.min().y - box_right.min().y) < 0.0)
                || (_axis == Z && (box_left.min().z - box_right.min().z) < 0.0)) {
                swap(e[j], e[j + 1]);
            }
        }
    }
}

class bvh_node : public hittable
{
public:
    __host__ __device__ bvh_node(hittable** src_objects, int start, int end, thrust::default_random_engine& rng, const char* name = nullptr);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;
    __host__ __device__ aabb bounding_box() const override;

    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ vector3 random(const vector3& o, thrust::default_random_engine& rng) const override;

    __host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableBVHNodeType; }

private:
    hittable* m_left = nullptr;
    hittable* m_right = nullptr;
    aabb m_bbox;

    __host__ __device__ static bool box_compare(const hittable* a, const hittable* b, int axis_index);
    __host__ __device__ static bool box_x_compare(const hittable* a, const hittable* b);
    __host__ __device__ static bool box_y_compare(const hittable* a, const hittable* b);
    __host__ __device__ static bool box_z_compare(const hittable* a, const hittable* b);

    // Utility function to swap hittable pointers
    __host__ __device__ static void swap(hittable** a, hittable** b);
};

__host__ __device__ inline bvh_node::bvh_node(hittable** src_objects, int start, int end, thrust::default_random_engine& rng, const char* name) {
    if (name != nullptr)
        setName(name);
    else
        setName("BVHNode");

    // Use Thrust random distribution for generating random axis
    thrust::uniform_int_distribution<int> axis_dist(0, 2);
    int axis = axis_dist(rng);

    auto comparator = (axis == 0) ? box_x_compare
        : (axis == 1) ? box_y_compare
        : box_z_compare;

    int object_span = end - start;

    if (object_span == 1) {
        m_left = m_right = src_objects[start];
    }
    else if (object_span == 2) {
        if (comparator(src_objects[start], src_objects[start + 1])) {
            m_left = src_objects[start];
            m_right = src_objects[start + 1];
        }
        else {
            m_left = src_objects[start + 1];
            m_right = src_objects[start];
        }
    }
    else {
        // Manual sort instead of thrust::sort
        for (size_t i = start; i < end; ++i) {
            for (size_t j = i + 1; j < end; ++j) {
                if (!comparator(src_objects[i], src_objects[j])) {
                    swap(&src_objects[i], &src_objects[j]);
                }
            }
        }

        int mid = start + object_span / 2;
        m_left = new bvh_node(src_objects, start, mid, rng, "left");
        m_right = new bvh_node(src_objects, mid, end, rng, "right");
    }

    m_bbox = aabb(m_left->bounding_box(), m_right->bounding_box());
}

__device__ bool inline bvh_node::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const {
    if (!m_bbox.hit(r, ray_t))
        return false;

    bool hit_left = m_left->hit(r, ray_t, rec, depth, max_depth, rng);
    bool hit_right = m_right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, depth, max_depth, rng);

    return hit_left || hit_right;
}

__device__ inline float bvh_node::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const {
    return 0.0;
}

__device__ inline vector3 bvh_node::random(const vector3& o, thrust::default_random_engine& rng) const {
    return vector3(1, 0, 0);
}

__host__ __device__ inline aabb bvh_node::bounding_box() const {
    return m_bbox;
}

__host__ __device__ inline bool bvh_node::box_compare(const hittable* a, const hittable* b, int axis_index) {
    return a->bounding_box().axis(axis_index).min < b->bounding_box().axis(axis_index).min;
}

__host__ __device__ inline bool bvh_node::box_x_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 0);
}

__host__ __device__ inline bool bvh_node::box_y_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 1);
}

__host__ __device__ inline bool bvh_node::box_z_compare(const hittable* a, const hittable* b) {
    return box_compare(a, b, 2);
}

__host__ __device__ inline void bvh_node::swap(hittable** a, hittable** b) {
    hittable* temp = *a;
    *a = *b;
    *b = temp;
}

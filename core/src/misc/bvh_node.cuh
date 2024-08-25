#pragma once

#include "ray.cuh"
#include "../primitives/hittable.cuh"
#include "../misc/gpu_randomizer.cuh"

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
            hittable *ah = e[j];
            hittable *bh = e[j+1];
            
            box_left = ah->bounding_box();
            box_right = bh->bounding_box();

            if ((_axis == X && (box_left.min().x - box_right.min().x) < 0.0)
             || (_axis == Y && (box_left.min().y - box_right.min().y) < 0.0)
             || (_axis == Z && (box_left.min().z - box_right.min().z) < 0.0)) {
                swap(e[j], e[j+1]);
            }
        }
    }
}

class bvh_node: public hittable
{
public:
    //__device__ bvh_node() {}


    //__device__ bvh_node(hittable **e, int n, float time0, float time1, curandState& local_rand_state, const char* name = nullptr);
    __device__ bvh_node(hittable** src_objects, int start, int end, curandState* local_rand_state, const char* name = nullptr);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;
    __host__ __device__ aabb bounding_box() const override;

    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;

    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    

    __host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableBVHNodeType; }



    //hittable *left;
    //hittable *right;
    //aabb box;

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

//__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& b) const
//{
//    b = m_box;
//    return true;
//}
//
//__device__ bool bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
//    if (box.hit(r, tmin, tmax)) {
//        hit_record left_rec, right_rec;
//        bool hit_left = left->hit(r, tmin, tmax, left_rec);
//        bool hit_right = right->hit(r, tmin, tmax, right_rec);
//        if (hit_left && hit_right) {
//            if (left_rec.t < right_rec.t) {
//                rec = left_rec;
//            } else {
//                rec = right_rec;
//            }
//            return true;
//        } else if (hit_left) {
//            rec = left_rec;
//            return true;
//        } else if (hit_right) {
//            rec = right_rec;
//            return true;
//        } else {
//            return false;
//        }
//    } else {
//        return false;
//    }
//}


// to remove old !!!!!!!!
//__device__ inline bvh_node::bvh_node(hittable **e, int n, float time0, float time1, curandState& local_rand_state, const char* name)
//{
//    if (name != nullptr)
//        setName(name);
//    else
//        setName("BVHNode");
//    
//    int axis = int(3 * curand_uniform(&local_rand_state));
//
//    if (axis == 0)
//        bubble_sort<X>(e, n);
//    else if (axis == 1)
//        bubble_sort<Y>(e, n);
//    else
//        bubble_sort<Z>(e, n);
//
//    if (n == 1) {
//        m_left = m_right = e[0];
//    }
//    else if (n == 2) {
//        m_left = e[0];
//        m_right = e[1];
//    }
//    else {
//        m_left = new bvh_node(e, n/2, time0, time1, local_rand_state);
//        m_right = new bvh_node(e + n/2, n - n/2, time0, time1, local_rand_state);
//    }
//
//    aabb box_left, box_right;
//
//    m_bbox = surrounding_box(box_left, box_right);
//}



// new one (mine)
__device__ inline bvh_node::bvh_node(hittable** src_objects, int start, int end, curandState* local_rand_state, const char* name)
{
    if (name != nullptr)
        setName(name);
    else
        setName("BVHNode");

    int axis = get_int(local_rand_state, 0, 2);
    auto comparator = (axis == 0) ? box_x_compare
        : (axis == 1) ? box_y_compare
        : box_z_compare;

    size_t object_span = end - start;

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

        auto mid = start + object_span / 2;
        m_left = new bvh_node(src_objects, start, mid, local_rand_state, "left");
        m_right = new bvh_node(src_objects, mid, end, local_rand_state, "right");
    }

    m_bbox = aabb(m_left->bounding_box(), m_right->bounding_box());
}

__device__ bool inline bvh_node::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    if (!m_bbox.hit(r, ray_t))
        return false;

    bool hit_left = m_left->hit(r, ray_t, rec, depth, local_rand_state);
    bool hit_right = m_right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec, depth, local_rand_state);

    return hit_left || hit_right;
}

__device__ inline float bvh_node::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0;
}

__device__ inline vector3 bvh_node::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3(1, 0, 0);
}

__host__ __device__ inline aabb bvh_node::bounding_box() const
{
    return m_bbox;
}



__host__ __device__ inline bool bvh_node::box_compare(const hittable* a, const hittable* b, int axis_index)
{
    return a->bounding_box().axis(axis_index).min < b->bounding_box().axis(axis_index).min;
}

__host__ __device__ inline bool bvh_node::box_x_compare(const hittable* a, const hittable* b)
{
    return box_compare(a, b, 0);
}

__host__ __device__ inline bool bvh_node::box_y_compare(const hittable* a, const hittable* b)
{
    return box_compare(a, b, 1);
}

__host__ __device__ inline bool bvh_node::box_z_compare(const hittable* a, const hittable* b)
{
    return box_compare(a, b, 2);
}

__host__ __device__ inline void bvh_node::swap(hittable** a, hittable** b)
{
    hittable* temp = *a;
    *a = *b;
    *b = temp;
}
#pragma once

#include "hittable.cuh"
#include "../utilities/interval.cuh"
#include "../misc/aabb.cuh"
#include "../misc/gpu_randomizer.cuh"



class hittable_list: public hittable
{
public:
    //__device__ hittable_list() {}
    //__device__ hittable_list(hittable **e, int n) { list = e; list_size = n; allocated_list_size = n; } 
    //__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    //__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    //__device__ void add(hittable* e);

    //hittable **list;
    //int list_size;
    //int allocated_list_size;

    hittable** objects;
    int object_count = 0;
    int object_capacity = 0;

    //__host__ __device__ hittable_list(const char* _name = nullptr);
    //__host__ __device__ hittable_list(hittable* object, const char* _name = nullptr);
    //__host__ __device__ hittable_list(hittable** l, int n, const char* _name = nullptr);

    __host__ __device__ hittable_list(const char* _name = nullptr) : objects(nullptr), object_count(0), object_capacity(0)
    {
        if (_name != nullptr)
            setName(_name);
        else
            setName("HittableList");
    }

    __host__ __device__ hittable_list(hittable* object, const char* _name = nullptr) : objects(nullptr), object_count(0), object_capacity(0)
    {
        //printf("hittable_list ctor %s\n", object ? object->getName() : "NULL");
        
        if (_name != nullptr)
            setName(_name);
        else
            setName("HittableList");

        add(object);
    }

    __host__ __device__ hittable_list(hittable** l, int n, const char* _name = nullptr) : objects(nullptr), object_count(0), object_capacity(0)
    {
        if (_name != nullptr)
            setName(_name);
        else
            setName("HittableList");

        for (int i = 0; i < n; i++)
        {
            add(l[i]);
        }
    }



    __host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableListType; }


    __host__ __device__ void clear();
    __host__ __device__ hittable* back();



    __host__ __device__ void add(hittable* object);
    __host__ __device__ hittable* get(const char* name);
    __host__ __device__ bool remove(hittable* object);



    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;
    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;


    /// <summary>
    /// Random special implementation for hittable list (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    __host__ __device__ aabb bounding_box() const override;
};





__host__ __device__ inline void hittable_list::clear()
{
    printf("clear %i objects\n", object_count);
    
    for (int i = 0; i < object_count; i++)
    {
        delete objects[i];
    }
    object_count = 0;
}


__host__ __device__ inline hittable* hittable_list::back()
{
    return objects[object_count - 1];
}


//__host__ __device__ inline void hittable_list::add(hittable* object)
//{
//    printf("add %s\n", object->getName());
//    
//    if (object_count == object_capacity)
//    {
//        // Increase the capacity
//        int new_capacity = object_capacity == 0 ? 1 : 2 * object_capacity;
//        hittable** new_objects = new hittable*[new_capacity];
//
//        // Copy the old objects to the new array
//        for (int i = 0; i < object_count; i++)
//        {
//            new_objects[i] = objects[i];
//        }
//
//        // Delete the old array
//        delete[] objects;
//
//        // Point to the new array
//        objects = new_objects;
//        object_capacity = new_capacity;
//    }
//
//    // Add the new object
//    //objects[object_count] = object;
//    //object_count++;
//
//    m_bbox = aabb(m_bbox, object->bounding_box());
//}

__host__ __device__ inline void hittable_list::add(hittable* e)
{
    //printf("add before %s %i/%i\n", e->getName(), object_count, object_capacity);

    if (e == nullptr || e->getName() == "")
    {
        printf("DIRTY DATA !!!!!\n");
        return;
    }

    if (object_capacity <= object_count)
    {
        hittable** new_list = new hittable*[object_count == 0 ? 2 : object_count * 2];
        for (int i = 0; i < object_count; i++) {
            new_list[i] = objects[i];
        }
        objects = new_list;
        object_capacity = object_count == 0 ? 2 : object_count * 2;
    }

    
    objects[object_count] = e;
    object_count++;

    printf("add to hittable_list %s %i/%i\n", e->getName(), object_count, object_capacity);

    m_bbox = aabb(m_bbox, e->bounding_box());
}

__host__ __device__ inline hittable* hittable_list::get(const char* name)
{
    for (int i = 0; i < object_count; i++)
    {
        if (objects[i]->getName() == name)
        {
            return objects[i];
        }
    }
    return nullptr;
}


__host__ __device__ inline bool hittable_list::remove(hittable* object)
{
    bool found = false;
    for (int i = 0; i < object_count; i++)
    {
        if (objects[i] == object)
        {
            found = true;
            delete objects[i];
            for (int j = i; j < object_count - 1; j++)
            {
                objects[j] = objects[j + 1];
            }
            object_count--;
            break;
        }
    }
    return found;
}

__device__ inline bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int i = 0; i < object_count; i++)
    {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec, depth, local_rand_state))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;


    //hit_record temp_rec;
    //bool hit_anything = false;
    //float closest_so_far = ray_t.max;

    //for (int i = 0; i < list_size; i++) {
    //    if (list[i]->hit(r, tmin, closest_so_far, temp_rec)) {
    //        hit_anything = true;
    //        closest_so_far = temp_rec.t;
    //        rec = temp_rec;
    //    }
    //}
    //return hit_anything;
}

__host__ __device__ inline aabb hittable_list::bounding_box() const
{
    return m_bbox;
}

__device__ inline float hittable_list::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    float weight = 1.0f / object_count;
    float sum = 0.0f;

    for (int i = 0; i < object_count; i++)
    {
        sum += weight * objects[i]->pdf_value(o, v, local_rand_state);
    }

    return sum;
}

//
///// <summary>
///// Random special implementation for hittable list (override base)
///// </summary>
///// <param name="origin"></param>
///// <returns></returns>
__device__ inline vector3 hittable_list::random(const vector3& o, curandState* local_rand_state) const
{
    if (object_count > 0)
    {
        return objects[get_int(local_rand_state, 0, object_count - 1)]->random(o, local_rand_state);
    }

    return vector3();
}


// old !!!!!!!!!!!
//__device__ bool hittable_list::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
//    hit_record temp_rec;
//    bool hit_anything = false;
//    float closest_so_far = tmax;
//
//    for (int i = 0; i < object_count; i++) {
//        if (objects[i]->hit(r, tmin, closest_so_far, temp_rec)) {
//            hit_anything = true;
//            closest_so_far = temp_rec.t;
//            rec = temp_rec;
//        }
//    }
//    return hit_anything;
//}

//__device__ bool hittable_list::bounding_box(float t0, float t1, aabb& box) const
//{
//    if (list_size < 1) {
//        return false;
//    }
//
//    aabb temp_box;
//    bool first_true = objects[0]->bounding_box(t0, t1, temp_box);
//    if (!first_true) {
//        return false;
//    } else {
//        box = temp_box;
//    }
//    
//    for (int i = 1; i < object_count; i++) {
//        if (objects[i]->bounding_box(t0, t1, temp_box)) {
//            box = surrounding_box(box, temp_box);
//        } else {
//            return false;
//        }
//    }
//    return true;
//}


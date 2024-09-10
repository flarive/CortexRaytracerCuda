#pragma once

#include "hittable.cuh"
#include "../utilities/interval.cuh"
#include "../misc/aabb.cuh"
#include "../misc/gpu_randomizer.cuh"

class hittable_list: public hittable
{
public:
    hittable** objects;
    unsigned int object_count = 0;
    unsigned int object_capacity = 0;

    __host__ __device__ hittable_list(const char* _name = nullptr) : objects(nullptr), object_count(0), object_capacity(0)
    {
        if (_name != nullptr)
            setName(_name);
        else
            setName("HittableList");
    }

    __host__ __device__ hittable_list(hittable* object, const char* _name = nullptr) : objects(nullptr), object_count(0), object_capacity(0)
    {
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

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableListType; }


    __host__ __device__ void clear();
    __host__ __device__ hittable* back();



    __host__ __device__ void add(hittable* object);
    __host__ __device__ hittable* get(const char* name);
    __host__ __device__ bool remove(hittable* object);



    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const override;


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
    
    for (unsigned int i = 0; i < object_count; i++)
    {
        delete objects[i];
    }
    object_count = 0;
}


__host__ __device__ inline hittable* hittable_list::back()
{
    return objects[object_count - 1];
}

__host__ __device__ inline void hittable_list::add(hittable* e)
{
    if (object_capacity <= object_count)
    {
        hittable** new_list = new hittable*[object_count == 0 ? 2 : object_count * 2];
        for (unsigned int i = 0; i < object_count; i++) {
            new_list[i] = objects[i];
        }
        objects = new_list;
        object_capacity = object_count == 0 ? 2 : object_count * 2;
    }

    
    objects[object_count] = e;
    object_count++;

    m_bbox = aabb(m_bbox, e->bounding_box());
}

__host__ __device__ inline hittable* hittable_list::get(const char* name)
{
    for (unsigned int i = 0; i < object_count; i++)
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
    for (unsigned int i = 0; i < object_count; i++)
    {
        if (objects[i] == object)
        {
            found = true;
            delete objects[i];
            for (unsigned int j = i; j < object_count - 1; j++)
            {
                objects[j] = objects[j + 1];
            }
            object_count--;
            break;
        }
    }
    return found;
}

__device__ inline bool hittable_list::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (unsigned int i = 0; i < object_count; i++)
    {
        if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec, depth, max_depth, local_rand_state))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

__host__ __device__ inline aabb hittable_list::bounding_box() const
{
    return m_bbox;
}

__device__ inline float hittable_list::pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const
{
    float weight = 1.0f / object_count;
    float sum = 0.0f;

    for (unsigned int i = 0; i < object_count; i++)
    {
        sum += weight * objects[i]->pdf_value(o, v, max_depth, local_rand_state);
    }

    return sum;
}

///// <summary>
///// Random special implementation for hittable list (override base)
///// </summary>
///// <param name="origin"></param>
///// <returns></returns>
__device__ inline vector3 hittable_list::random(const vector3& o, curandState* local_rand_state) const
{
    if (object_count > 0)
    {
        unsigned int r = 0;
        
        if (object_count > 1)
            r = get_int(local_rand_state, 0, object_count - 1);

        if (r >= object_count)
        {
            printf("Error: index out of bounds! lll = %d, object_count = %d\n", r, object_count);
            return vector3(); // Or handle the error accordingly
        }

        if (!objects[r])
        {
            printf("Error: nullptr encountered at objects[%d]!\n", r);
            return vector3(); // Or handle the error accordingly
        }

        return objects[r]->random(o, local_rand_state);
    }

    return vector3();
}
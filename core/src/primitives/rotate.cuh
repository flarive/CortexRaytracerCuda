#pragma once

//#define DEGREES_TO_RADIANS(degrees)((M_PI * degrees)/180)

#include "hittable.cuh"

namespace rt
{
    class rotate : public hittable
    {
    public:
        //__device__ rotate(hittable* p, float angle);

        //__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        //__device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const {
        //    output_box = bbox;
        //    return hasbox;
        //}

        //hittable* ptr;
        //float sin_theta;
        //float cos_theta;
        //bool hasbox;
        //aabb bbox;

        __host__ __device__ rotate(hittable* _p, const vector3& _rotation);

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;
        __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;
        __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;
        __host__ __device__ aabb bounding_box() const override;

        __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTransformRotateType; }


    private:
        //std::shared_ptr<hittable> m_object;
        //double m_sin_theta = 0;
        //double m_cos_theta = 0;
        //int m_axis = 0;



        //glm::dquat m_rotationQuaternion;
        //vector3 m_center;
        //vector3 m_halfExtents;

        //point3 m_center;


        hittable* m_object;
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        aabb bbox;

        vector3 m_rotation{};
    };
}


__host__ __device__ rt::rotate::rotate(hittable* _object, const vector3& _rotation) : m_object(_object), m_rotation(_rotation)
{
    m_name = _object->getName();

    auto radians_x = degrees_to_radians(_rotation.x);
    auto radians_y = degrees_to_radians(_rotation.y);
    auto radians_z = degrees_to_radians(_rotation.z);


    matrix4 rotationMatrix(1.0f);
    rotationMatrix = glm::rotate(rotationMatrix, radians_x, vector3(1.0f, 0.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, radians_y, vector3(0.0f, 1.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, radians_z, vector3(0.0f, 0.0f, 1.0f));

    bbox = m_object->bounding_box();

    point3 min(INFINITY, INFINITY, INFINITY);
    point3 max(-INFINITY, -INFINITY, -INFINITY);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                vector4 corner(
                    i * bbox.x.max + (1 - i) * bbox.x.min,
                    j * bbox.y.max + (1 - j) * bbox.y.min,
                    k * bbox.z.max + (1 - k) * bbox.z.min,
                    1.0f
                );

                vector4 rotatedCorner = rotationMatrix * corner;
                vector3 tester(rotatedCorner.x, rotatedCorner.y, rotatedCorner.z);

                for (int c = 0; c < 3; c++) {
                    min[c] = fmin(min[c], tester[c]);
                    max[c] = fmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ bool rt::rotate::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
    // Change the ray from world space to object space
    auto origin = r.origin();
    auto direction = r.direction();

    vector4 origin_vec(origin[0], origin[1], origin[2], 1.0f);
    vector4 direction_vec(direction[0], direction[1], direction[2], 0.0f);

    auto radians_x = degrees_to_radians(-m_rotation.x);
    auto radians_y = degrees_to_radians(-m_rotation.y);
    auto radians_z = degrees_to_radians(-m_rotation.z);

    matrix4 inverseRotationMatrix(1.0f);
    inverseRotationMatrix = glm::rotate(inverseRotationMatrix, radians_x, vector3(1.0f, 0.0f, 0.0f));
    inverseRotationMatrix = glm::rotate(inverseRotationMatrix, radians_y, vector3(0.0f, 1.0f, 0.0f));
    inverseRotationMatrix = glm::rotate(inverseRotationMatrix, radians_z, vector3(0.0f, 0.0f, 1.0f));

    vector4 rotated_origin = inverseRotationMatrix * origin_vec;
    vector4 rotated_direction = inverseRotationMatrix * direction_vec;

    ray rotated_r(point3(rotated_origin.x, rotated_origin.y, rotated_origin.z),
        vector3(rotated_direction.x, rotated_direction.y, rotated_direction.z),
        r.time());

    // Determine whether an intersection exists in object space (and if so, where)
    if (!m_object->hit(rotated_r, ray_t, rec, depth, max_depth, local_rand_state))
        return false;

    // Change the intersection point from object space to world space
    vector4 hit_point_vec(rec.hit_point[0], rec.hit_point[1], rec.hit_point[2], 1.0f);
    vector4 normal_vec(rec.normal[0], rec.normal[1], rec.normal[2], 0.0f);

    matrix4 rotationMatrix(1.0f);
    rotationMatrix = glm::rotate(rotationMatrix, degrees_to_radians(m_rotation.x), vector3(1.0f, 0.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, degrees_to_radians(m_rotation.y), vector3(0.0f, 1.0f, 0.0f));
    rotationMatrix = glm::rotate(rotationMatrix, degrees_to_radians(m_rotation.z), vector3(0.0f, 0.0f, 1.0f));

    vector4 world_hit_point = rotationMatrix * hit_point_vec;
    vector4 world_normal = rotationMatrix * normal_vec;

    rec.hit_point = point3(world_hit_point.x, world_hit_point.y, world_hit_point.z);
    rec.normal = vector3(world_normal.x, world_normal.y, world_normal.z);

    return true;
}

__device__ float  rt::rotate::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 rt::rotate::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3();
}

__host__ __device__ aabb rt::rotate::bounding_box() const
{
    return bbox;
}


//
//__device__ rotate::rotate(hittable *p, float angle) : ptr(p) {
//    auto radians = DEGREES_TO_RADIANS(angle);
//    sin_theta = sin(radians);
//    cos_theta = cos(radians);
//    hasbox = ptr->bounding_box(0, 1, bbox);
//
//    vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
//    vector3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
//
//    for (int i = 0; i < 2; i++) {
//        for (int j = 0; j < 2; j++) {
//            for (int k = 0; k < 2; k++) {
//                auto x = i*bbox.max().x + (1-i)*bbox.min().x;
//                auto y = j*bbox.max().y + (1-j)*bbox.min().y;
//                auto z = k*bbox.max().z + (1-k)*bbox.min().z;
//
//                auto newx =  cos_theta*x + sin_theta*z;
//                auto newz = -sin_theta*x + cos_theta*z;
//
//                vector3 tester(newx, y, newz);
//
//                for (int c = 0; c < 3; c++) {
//                    min[c] = ffmin(min[c], tester[c]);
//                    max[c] = ffmax(max[c], tester[c]);
//                }
//            }
//        }
//    }
//
//    bbox = aabb(min, max);
//}
//
//__device__ bool rotate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//    vector3 origin = r.origin();
//    vector3 direction = r.direction();
//
//    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
//    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];
//
//    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
//    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];
//
//    ray rotated_r(origin, direction, r.time());
//
//    if (!ptr->hit(rotated_r, t_min, t_max, rec))
//        return false;
//
//    vector3 p = rec.p;
//    vector3 normal = rec.normal;
//
//    p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
//    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];
//
//    normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
//    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];
//
//    rec.p = p;
//    rec.set_face_normal(rotated_r, normal);
//
//    return true;
//}
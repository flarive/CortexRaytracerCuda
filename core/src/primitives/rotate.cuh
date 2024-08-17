#pragma once

#define DEGREES_TO_RADIANS(degrees)((M_PI * degrees)/180)

#include "hittable.cuh"
#include "../misc/constants.cuh"


class rotate : public hittable
{
public:
    __device__ rotate(hittable* p, float angle);

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const {
        output_box = bbox;
        return hasbox;
    }

    hittable* ptr;
    float sin_theta;
    float cos_theta;
    bool hasbox;
    aabb bbox;
};

__device__ rotate::rotate(hittable *p, float angle) : ptr(p) {
    auto radians = DEGREES_TO_RADIANS(angle);
    sin_theta = sin(radians);
    cos_theta = cos(radians);
    hasbox = ptr->bounding_box(0, 1, bbox);

    vector3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    vector3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                auto x = i*bbox.max().x + (1-i)*bbox.min().x;
                auto y = j*bbox.max().y + (1-j)*bbox.min().y;
                auto z = k*bbox.max().z + (1-k)*bbox.min().z;

                auto newx =  cos_theta*x + sin_theta*z;
                auto newz = -sin_theta*x + cos_theta*z;

                vector3 tester(newx, y, newz);

                for (int c = 0; c < 3; c++) {
                    min[c] = ffmin(min[c], tester[c]);
                    max[c] = ffmax(max[c], tester[c]);
                }
            }
        }
    }

    bbox = aabb(min, max);
}

__device__ bool rotate::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
    vector3 origin = r.origin();
    vector3 direction = r.direction();

    origin[0] = cos_theta*r.origin()[0] - sin_theta*r.origin()[2];
    origin[2] = sin_theta*r.origin()[0] + cos_theta*r.origin()[2];

    direction[0] = cos_theta*r.direction()[0] - sin_theta*r.direction()[2];
    direction[2] = sin_theta*r.direction()[0] + cos_theta*r.direction()[2];

    ray rotated_r(origin, direction, r.time());

    if (!ptr->hit(rotated_r, t_min, t_max, rec))
        return false;

    vector3 p = rec.p;
    vector3 normal = rec.normal;

    p[0] =  cos_theta*rec.p[0] + sin_theta*rec.p[2];
    p[2] = -sin_theta*rec.p[0] + cos_theta*rec.p[2];

    normal[0] =  cos_theta*rec.normal[0] + sin_theta*rec.normal[2];
    normal[2] = -sin_theta*rec.normal[0] + cos_theta*rec.normal[2];

    rec.p = p;
    rec.set_face_normal(rotated_r, normal);

    return true;
}
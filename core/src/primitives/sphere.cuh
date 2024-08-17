#pragma once

#include "../misc/constants.cuh"
#include "hittable.cuh"
#include "../materials/material.cuh"

class sphere: public hittable
{
public:
    //sphere() {}
    __device__ sphere(vector3 cen, float r, material* m): center(cen), radius(r), mat_ptr(m) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    __device__ void get_sphere_uv(const vector3& p, float& u, float& v) const;
    vector3 center;
    float radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
    vector3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(b * b - a * c)) / a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.point(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
            return true;
        }
        temp = (-b + sqrt(b*b - a*c)) / a;
        if (temp < tmax && temp > tmin) {
            rec.t = temp;
            rec.p = r.point(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
            return true;
        }
    }
    return false;
}

__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {
    box = aabb(center - vector3(radius, radius, radius),
               center + vector3(radius, radius, radius));
    return true;
}

__device__ void sphere::get_sphere_uv(const vector3& p, float& u, float& v) const {
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    u = 1-(phi + M_PI) / (2*M_PI);
    v = (theta + M_PI/2) / M_PI;
}
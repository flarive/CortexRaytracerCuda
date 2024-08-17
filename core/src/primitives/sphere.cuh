#pragma once

#include "../misc/constants.cuh"
#include "entity.cuh"
#include "../materials/material.cuh"

class Sphere: public Entity {
public:
    //Sphere() {}
    __device__ Sphere(vector3 cen, float r, Material* m): center(cen), radius(r), mat_ptr(m) {};
    __device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    __device__ void get_sphere_uv(const vector3& p, float& u, float& v) const;
    vector3 center;
    float radius;
    Material* mat_ptr;
};

__device__ bool Sphere::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {
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

__device__ bool Sphere::bounding_box(float t0, float t1, aabb& box) const {
    box = aabb(center - vector3(radius, radius, radius),
               center + vector3(radius, radius, radius));
    return true;
}

__device__ void Sphere::get_sphere_uv(const vector3& p, float& u, float& v) const {
    float phi = atan2(p.z, p.x);
    float theta = asin(p.y);
    u = 1-(phi + M_PI) / (2*M_PI);
    v = (theta + M_PI/2) / M_PI;
}
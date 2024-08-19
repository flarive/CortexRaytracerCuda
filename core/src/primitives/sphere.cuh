#pragma once

#include "../misc/constants.cuh"
#include "hittable.cuh"
#include "../materials/material.cuh"
#include "../misc/onb.cuh"

class sphere: public hittable
{
public:
    //__device__ sphere(vector3 cen, float r, material* m): center(cen), radius(r), mat_ptr(m) {};
    //__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
    //__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const;
    //__device__ void get_sphere_uv(const vector3& p, float& u, float& v) const;
    //vector3 center;
    //float radius;
    //material* mat_ptr;

    __host__ __device__ sphere(point3 _center, float _radius, const char* _name)
        : sphere(_center, _radius, nullptr, uvmapping(), _name)
    {
        is_moving = false;
    }


    __host__ __device__ sphere(point3 _center, float _radius, material* _material, const char* _name)
        : sphere(_center, _radius, _material, uvmapping(), _name)
    {
        is_moving = false;
    }

    __host__ __device__ sphere(point3 _center1, point3 _center2, float _radius, material* _material, const char* _name)
        : center1(_center1), radius(_radius), mat(_material), is_moving(true)
    {
        if (_name != nullptr)
            setName(_name);
        else
            setName(new char[7] {"Sphere"});

        // calculate moving sphere bounding box for ray optimizations
        vector3 rvec = vector3(radius, radius, radius);
        aabb box1(_center1 - rvec, _center1 + rvec);
        aabb box2(_center2 - rvec, _center2 + rvec);
        m_bbox = aabb(box1, box2);

        center_vec = _center2 - _center1;
    }

    __host__ __device__ sphere(point3 _center, float _radius, material* _material, const uvmapping& _mapping, const char* _name)
        : center1(_center), radius(_radius), mat(_material), is_moving(false)
    {
        if (_name != nullptr)
            setName(_name);
        else
            setName(new char[7] {"Sphere"});

        m_mapping = _mapping;

        // calculate stationary sphere bounding box for ray optimizations
        vector3 rvec = vector3(_radius, _radius, _radius);
        m_bbox = aabb(_center - rvec, _center + rvec);
    }


    __host__ __device__ aabb bounding_box() const override;

    __host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableSphereType; }

    /// <summary>
    /// Logic of sphere ray hit detection
    /// </summary>
    /// <param name="r"></param>
    /// <param name="ray_t"></param>
    /// <param name="rec"></param>
    /// <returns></returns>
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;



    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;


    /// <summary>
    /// Random special implementation for sphere (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const point3& o, curandState* local_rand_state) const override;


private:
    point3 center1{};
    float radius = 0;
    material* mat;
    bool is_moving = false;
    vector3 center_vec{};

    __host__ __device__ point3 sphere_center(float time) const;


    __host__ __device__ static void getTangentAndBitangentAroundPoint(const vector3& p, float radius, float phi, float theta, vector3& tan, vector3& bitan);

};



__host__ __device__ aabb sphere::bounding_box() const
{
    return m_bbox;
}

/// <summary>
/// Logic of sphere ray hit detection
/// </summary>
/// <param name="r"></param>
/// <param name="ray_t"></param>
/// <param name="rec"></param>
/// <returns></returns>
__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    //glm::vec3 local_v(1.0f, 2.0f, 3.0f);
    //glm::vec3 local_v2(1.0f, 2.0f, 3.0f);

    //auto kkk = glm::dot(local_v, local_v2);



    
    point3 center = is_moving ? sphere_center(r.time()) : center1;
    vector3 oc = r.origin() - center;
    float a = vector_length_squared(r.direction());
    float half_b = dot(oc, r.direction());
    float c = vector_length_squared(oc) - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    float sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (-half_b - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (-half_b + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    // number of hits encountered by the ray (only the nearest ?)
    rec.t = root;

    // point coordinate of the hit
    rec.hit_point = r.at(rec.t);

    // material of the hit object
    rec.mat = mat;

    // name of the primitive hit by the ray
    rec.name = m_name;
    rec.bbox = m_bbox;

    // set normal and front-face tracking
    vector3 outward_normal = (rec.hit_point - center) / radius;
    rec.set_face_normal(r, outward_normal);

    // compute phi and theta for tangent and bitangent calculation
    float phi = atan2(outward_normal.z, outward_normal.x);
    float theta = acos(outward_normal.y);

    // compute sphere primitive tangent and bitangent for normals
    vector3 tan, bitan;
    getTangentAndBitangentAroundPoint(outward_normal, radius, phi, theta, tan, bitan);

    // store tangents and bitangents in the hit record if needed
    rec.tangent = tan;
    rec.bitangent = bitan;


    // compute UV coordinates
    get_sphere_uv(outward_normal, rec.u, rec.v, m_mapping);

    return true;
}

__device__ float sphere::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    // This method only works for stationary spheres.

    hit_record rec;
    if (!this->hit(ray(o, v), interval(get_shadow_acne_fix(), get_infinity()), rec, 0, local_rand_state))
        return 0;

    auto cos_theta_max = sqrt(1 - radius * radius / vector_length_squared(center1 - o));
    auto solid_angle = 2 * get_pi() * (1 - cos_theta_max);

    return  1 / solid_angle;
}

/// <summary>
/// Random special implementation for sphere (override base)
/// </summary>
/// <param name="origin"></param>
/// <returns></returns>
__device__ vector3 sphere::random(const point3& o, curandState* local_rand_state) const
{
    vector3 direction = center1 - o;
    auto distance_squared = vector_length_squared(direction);
    onb uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(local_rand_state, radius, distance_squared));
}



__host__ __device__ point3 sphere::sphere_center(float time) const
{
    // Linearly interpolate from center1 to center2 according to time, where t=0 yields
    // center1, and t=1 yields center2.
    return center1 + time * center_vec;
}

/// <summary>
/// https://medium.com/@dbildibay/ray-tracing-adventure-part-iv-678768947371
/// </summary>
/// <param name="p"></param>
/// <param name="radius"></param>
/// <param name="phi"></param>
/// <param name="theta"></param>
/// <param name="tan"></param>
/// <param name="bitan"></param>
__host__ __device__ void sphere::getTangentAndBitangentAroundPoint(const vector3& p, float radius, float phi, float theta, vector3& tan, vector3& bitan)
{
    // Tangent in the direction of increasing phi
    //tan.x = -p.z;
    //tan.y = 0;
    //tan.z = p.x;

    // hack FL to try having same normal rendering between sphere and obj
    tan.x = p.z;
    tan.y = 0;
    tan.z = -p.x;

    // Normalize the tangent
    tan = glm::normalize(tan);

    // Bitangent in the direction of increasing theta
    bitan = glm::cross(p, tan);

    // Ensure bitangent is perpendicular to both tangent and normal
    bitan = glm::normalize(bitan);
}



//__device__ bool sphere::hit(const ray& r, float tmin, float tmax, hit_record& rec) const {
//    vector3 oc = r.origin() - center;
//    float a = dot(r.direction(), r.direction());
//    float b = dot(oc, r.direction());
//    float c = dot(oc, oc) - radius * radius;
//    float discriminant = b*b - a*c;
//    if (discriminant > 0) {
//        float temp = (-b - sqrt(b * b - a * c)) / a;
//        if (temp < tmax && temp > tmin) {
//            rec.t = temp;
//            rec.p = r.point(rec.t);
//            rec.normal = (rec.p - center) / radius;
//            rec.mat_ptr = mat_ptr;
//            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
//            return true;
//        }
//        temp = (-b + sqrt(b*b - a*c)) / a;
//        if (temp < tmax && temp > tmin) {
//            rec.t = temp;
//            rec.p = r.point(rec.t);
//            rec.normal = (rec.p - center) / radius;
//            rec.mat_ptr = mat_ptr;
//            get_sphere_uv((rec.p-center)/radius, rec.u, rec.v);
//            return true;
//        }
//    }
//    return false;
//}
//
//__device__ bool sphere::bounding_box(float t0, float t1, aabb& box) const {
//    box = aabb(center - vector3(radius, radius, radius),
//               center + vector3(radius, radius, radius));
//    return true;
//}
//
//__device__ void sphere::get_sphere_uv(const vector3& p, float& u, float& v) const {
//    float phi = atan2(p.z, p.x);
//    float theta = asin(p.y);
//    u = 1-(phi + M_PI) / (2*M_PI);
//    v = (theta + M_PI/2) / M_PI;
//}
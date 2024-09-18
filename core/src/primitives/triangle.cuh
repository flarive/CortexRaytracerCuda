#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"

#include <cmath>

/// <summary>
/// Triangle primitive
/// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/triangle.h
/// </summary>
class triangle : public hittable
{
public:
    __host__ __device__ triangle(const char* _name = "Triangle");
    __host__ __device__ triangle(const vector3 v0, const vector3 v1, const vector3 v2, material* m, const char* _name = "Triangle");
    __host__ __device__ triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0, const vector3 vn1, const vector3 vn2, bool smooth_shading, material* m, const char* _name = "Triangle");

    __host__ __device__ triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0, const vector3 vn1, const vector3 vn2, const vector2& vuv0, const vector2& vuv1, const vector2& vuv2, bool smooth_shading, material* m, const char* _name = "Triangle");

    __host__ __device__ triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0,
        const vector3 vn1, const vector3 vn2,
        const vector2& vuv0, const vector2& vuv1, const vector2& vuv2,
        const vector3& tan0, const vector3& tan1, const vector3& tan2,
        const vector3& bitan0, const vector3& bitan1, const vector3& bitan2,
        bool smooth_shading, material* m, const char* _name = "Triangle");



    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTriangleType; }

    __host__ __device__ virtual aabb bounding_box() const override;


    /// <summary>
    /// Random special implementation for sphere (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const point3& o, thrust::default_random_engine& rng) const override;

public:
    vector3 verts[3]{};
    vector3 vert_normals[3]{};
    vector2 vert_uvs[3]{};

    vector3 vert_tangents[3]{};
    vector3 vert_bitangents[3]{};

    bool smooth_normals = false;
    material* mat_ptr = nullptr;
private:
    float area = 0.0f;
    vector3 middle_normal{};

    vector3 v0_v1{};
    vector3 v0_v2{};
};



#define EPS 0.000001f

__host__ __device__ inline triangle::triangle(const char* _name)
{
}

__host__ __device__ inline triangle::triangle(const vector3 v0, const vector3 v1, const vector3 v2, material* m, const char* _name)
    : triangle(v0, v1, v2, vector3(), vector3(), vector3(), vector2(), vector2(), vector2(), vector3(), vector3(), vector3(), vector3(), vector3(), vector3(), false, m, _name)
{
}

__host__ __device__ inline triangle::triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0, const vector3 vn1, const vector3 vn2, bool smooth_shading, material* m, const char* _name)
    : triangle(v0, v1, v2, vn0, vn1, vn2, vector2(), vector2(), vector2(), vector3(), vector3(), vector3(), vector3(), vector3(), vector3(), false, m, _name)
{
}

__host__ __device__ inline triangle::triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0, const vector3 vn1, const vector3 vn2, const vector2& vuv0, const vector2& vuv1, const vector2& vuv2, bool smooth_shading, material* m, const char* _name)
    : triangle(v0, v1, v2, vn0, vn1, vn2, vuv0, vuv1, vuv2, vector3(), vector3(), vector3(), vector3(), vector3(), vector3(), false, m, _name)
{
}

__host__ __device__ inline triangle::triangle(const vector3 v0, const vector3 v1, const vector3 v2, const vector3 vn0, const vector3 vn1, const vector3 vn2, const vector2& vuv0, const vector2& vuv1, const vector2& vuv2, const vector3& tan0, const vector3& tan1, const vector3& tan2,
    const vector3& bitan0, const vector3& bitan1, const vector3& bitan2, bool smooth_shading, material* m, const char* _name) : mat_ptr(m)
{
    verts[0] = v0;
    verts[1] = v1;
    verts[2] = v2;

    vert_normals[0] = unit_vector(vn0);
    vert_normals[1] = unit_vector(vn1);
    vert_normals[2] = unit_vector(vn2);

    vert_uvs[0] = vuv0;
    vert_uvs[1] = vuv1;
    vert_uvs[2] = vuv2;

    vert_tangents[0] = tan0;
    vert_tangents[1] = tan1;
    vert_tangents[2] = tan2;

    vert_bitangents[0] = bitan0;
    vert_bitangents[1] = bitan1;
    vert_bitangents[2] = bitan2;

    smooth_normals = smooth_shading;

    v0_v1 = verts[1] - verts[0];
    v0_v2 = verts[2] - verts[0];

    auto a = (v0 - v1).length();
    auto b = (v1 - v2).length();
    auto c = (v2 - v0).length();
    float s = (a + b + c) / 2.0f;
    area = glm::sqrt(fabsf(s * (s - a) * (s - b) * (s - c)));
    middle_normal = unit_vector(glm::cross(v0 - v1, v0 - v2));

    // bounding box
    vector3 max_extent = max(max(verts[0], verts[1]), verts[2]);
    vector3 min_extent = min(min(verts[0], verts[1]), verts[2]);
    float eps = 0.001f;
    auto epsv = vector3(eps, eps, eps);
    m_bbox = aabb(min_extent - epsv, max_extent + epsv);
}

__device__ inline bool triangle::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
    // Möller-Trumbore algorithm for fast triangle hit
    // https://web.archive.org/web/20200927071045/https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
    /*auto v0_v1 = verts[1] - verts[0];
    auto v0_v2 = verts[2] - verts[0];*/
    auto dir = r.direction();
    auto parallel_vec = glm::cross(dir, v0_v2);
    auto det = glm::dot(v0_v1, parallel_vec);
    // If det < 0, this is a back-facing intersection, change hit_record front_face
    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < EPS) return false;
    float inv_det = 1.0f / det;




    auto tvec = r.origin() - verts[0];
    auto u = glm::dot(tvec, parallel_vec) * inv_det;
    if (u < 0 || u > 1) return false;

    auto qvec = glm::cross(tvec, v0_v1);
    auto v = dot(dir, qvec) * inv_det;
    if (v < 0 || u + v > 1) return false;

    float t = glm::dot(v0_v2, qvec) * inv_det;

    if (t < ray_t.min || t > ray_t.max) return false;

    rec.t = t;

    // UV coordinates
    get_triangle_uv(rec.hit_point, rec.u, rec.v, verts, vert_uvs);

    rec.hit_point = r.at(t);
    rec.mat = mat_ptr;

    vector3 normal = middle_normal;

    if (smooth_normals)
    {
        float a = u, b = v, c = 1 - u - v;
        // What does u and v map to?
        normal = a * vert_normals[1] + b * vert_normals[2] + c * vert_normals[0];
    }

    // set normal and front-face tracking
    vector3 outward_normal = normal;
    rec.set_face_normal(r, outward_normal);

    // no need to calculate tangents and bitangents, just get them from obj file
    rec.tangent = vert_tangents[0];
    rec.bitangent = vert_bitangents[0];

    return true;
}

__host__ __device__ inline aabb triangle::bounding_box() const
{
    return m_bbox;
}

__device__ inline float triangle::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
    hit_record rec;
    if (!this->hit(ray(o, v), interval(EPS, INFINITY), rec, 0, max_depth, rng))
        return 0;

    // from https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4121581
    vector3 R1 = verts[0] - o, R2 = verts[1] - o, R3 = verts[2] - o;
    int r1 = R1.length();
    int r2 = R2.length();
    int r3 = R3.length();
    float N = glm::dot(R1, cross(R2, R3));
    float D = r1 * r2 * r3 + glm::dot(R1, R2) * r3 + glm::dot(R1, R3) * r2 + glm::dot(R2, R3) * r3;

    float omega = atan2(N, D);

    return 1.0f / omega;
}

__device__ inline vector3 triangle::random(const point3& o, thrust::default_random_engine& rng) const
{
    // From https://math.stackexchange.com/questions/18686/uniform-random-point-in-triangle-in-3d
    float r1 = get_real(rng);
    float r2 = get_real(rng);
    float ca = (1.0f - glm::sqrt(r1)), cb = glm::sqrt(r1) * (1.0f - r2), cc = r2 * glm::sqrt(r1);
    vector3 random_in_triangle = verts[0] * ca + verts[1] * cb + verts[2] * cc;
    return random_in_triangle - o;
}
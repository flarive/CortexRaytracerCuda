#pragma once

#include "hittable.cuh"
#include "../materials/material.cuh"

#include <algorithm>
#include <cmath>

/// <summary>
/// https://github.com/kamiyo/RayTra/blob/master/RayTra/Torus.cpp
/// </summary>
class torus : public hittable
{
public:
	__host__ __device__ torus(const char* _name = "Torus");
	__host__ __device__ torus(point3 _center, float _majorRadius, float _minorRadius, material* _material, const char* _name = "Torus");
	__host__ __device__ torus(point3 _center, float _majorRadius, float _minorRadius, material* _material, const uvmapping& _mapping, const char* _name = "Torus");

	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const override;

    __device__ vector3 random(const vector3& o, thrust::default_random_engine& rng) const override;

	__host__ __device__ aabb bounding_box() const override;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::hittableTorusType; }

private:
	point3 center{};
	float majorRadius = 0.0f;
	float minorRadius = 0.0f;
	material* mat = nullptr;
	float _R2 = 0.0f, _R2r2 = 0.0f;
};


__host__ __device__ torus::torus(const char* _name) : torus(vector3(0, 0, 0), 0.5, 0.25, nullptr, uvmapping(), _name)
{
}

__host__ __device__ torus::torus(point3 _center, float _majorRadius, float _minorRadius, material* _material, const char* _name)
	: torus(_center, _majorRadius, _minorRadius, _material, uvmapping(), _name)
{
}

__host__ __device__ torus::torus(point3 _center, float _majorRadius, float _minorRadius, material* _material, const uvmapping& _mapping, const char* _name)
	: center(_center), majorRadius(_majorRadius), minorRadius(_minorRadius), mat(_material)
{
	setName(_name);
	m_mapping = _mapping;

	// calculate torus bounding box for ray optimizations
	float rR = minorRadius + majorRadius;
	m_bbox = aabb(center + point3(-rR, -rR, -minorRadius), center + point3(rR, rR, minorRadius));

	_R2 = majorRadius * majorRadius;
	_R2r2 = _R2 - (minorRadius * minorRadius);
}


__device__ bool torus::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, thrust::default_random_engine& rng) const
{
	//const vector3 d = r.direction();
	//const vector3 e = r.origin() - center;

	//float dx2 = d.x * d.x, dy2 = d.y * d.y;
	//float ex2 = e.x * e.x, ey2 = e.y * e.y;
	//float dxex = d.x * e.x, dyey = d.y * e.y;

	//float A = glm::dot(d, d);
	//float B = 2 * glm::dot(d, e);
	//float C = glm::dot(e, e) + (_R2r2);
	//float D = 4 * _R2 * (dx2 + dy2);
	//float E = 8 * _R2 * (dxex + dyey);
	//float F = 4 * _R2 * (ex2 + ey2);

	//Eigen::VectorXf op(5);
	//op << C * C - F, 2 * B * C - E, 2 * A * C + B * B - D, 2 * A * B, A* A;

	//Eigen::PolynomialSolver<float, 4> psolve(op);
	//std::vector<float> reals;
	//psolve.realRoots(reals);

	//for (int i = static_cast<int>(reals.size()) - 1; i >= 0; i--)
	//{
	//	if (reals[i] < ray_t.min || reals[i] > ray_t.max)
	//	{
	//		reals.erase(reals.begin() + i);
	//	}
	//}

	//if (reals.empty())
	//{
	//	return false;
	//}

	//std::sort(reals.begin(), reals.end());
	//rec.t = reals[0];
	//vector3 p = e + rec.t * d;

	//vector3 pp = vector3(p.x, p.y, 0.);
	//vector3 c = glm::normalize(pp) * majorRadius; // center of tube
	//vector3 n = glm::normalize(p - c);

	//rec.hit_point = r.at(rec.t);

	//// Set normal and front-face tracking
	//vector3 outward_normal = glm::normalize(rec.hit_point - c);
	//rec.set_face_normal(r, outward_normal);

	// UV coordinates
	//float u, v;
	//get_torus_uv(p, c, u, v, majorRadius, minorRadius, m_mapping);

	//// Set UV coordinates
	//rec.u = u;
	//rec.v = v;

	rec.mat = mat;
	rec.name = m_name;
	rec.bbox = m_bbox;

	return true;
}


__device__ float torus::pdf_value(const point3& o, const vector3& v, int max_depth, thrust::default_random_engine& rng) const
{
	return 0.0f;
}


__device__ vector3 torus::random(const vector3& o, thrust::default_random_engine& rng) const
{
    return vector3();
}


__host__ __device__ aabb torus::bounding_box() const
{
	return m_bbox;
}
#pragma once

#include "../misc/vector3.cuh"
#include "../misc/aabb.cuh"
#include "../misc/ray.cuh"

class material;

struct hit_record
{
    //float t;
    //vector3 p;
    //vector3 normal;
    //material* mat_ptr;
    //float u;
    //float v;
    //bool front_face;

	point3 hit_point{}; // point (coordinates) where the hit occurred
	vector3 normal{}; // normal vector where the hit occurred
	material* mat = nullptr; // material of the object hit by the ray

	/// <summary>
	/// The t value in the hit_record specifies the distance from the ray origin A to the intersection point along the direction B.
	/// When a ray intersects an object, the t value is used to compute the exact position of the hit point :
	/// </summary>
	float t = 0.0f; // scalar parameter that moves the point along the ray.


	float u = 0.0f; // u mapping coordinate
	float v = 0.0f; // v mapping coordinate
	bool front_face = true; // front-face tracking (object was hit from outside (frontface) or inside (backface) ?)
	char* name = nullptr; // name of the object that was hit
	aabb bbox; // bounding box size of the object that was hit

	vector3 tangent{}; // tangent vector calculated from the normal (obj models only)
	vector3 bitangent{}; // bitangent vector calculated from the normal (obj models only)

	__device__ inline void set_face_normal(const ray& r, const vector3& outward_normal)
    {
		front_face = dot(r.direction(), outward_normal) < 0.0f;
        normal = front_face ? outward_normal : -outward_normal;
    }
};
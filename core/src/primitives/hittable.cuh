#pragma once

#include "../misc/aabb.cuh"
#include "../misc/ray.cuh"
#include "../misc/hit_record.cuh"
#include "../utilities/uvmapping.cuh"


__host__ __device__ enum class HittableTypeID {
	hittableBaseType = 0,
	hittableListType = 1,
	hittableBVHNodeType = 2,
	hittableAaRectType = 3,
	hittableBoxType = 4,
	hittableConeType = 5,
	hittableCylinderType = 6,
	hittableDiskType = 7,
	hittableQuadType = 8,
	hittableSphereType = 9,
	hittableTorusType = 10,
	hittableTriangleType = 11,
	hittableVolumeType = 12,
	hittableTransformFlipNormalType = 13,
	hittableTransformRotateType = 14,
	hittableTransformTranslateType = 15,
	hittableTransformScaleType = 16,
	lightType = 17,
	lightOmniType = 18,
	lightDirectionalType = 19,
	lightSpotType = 20
};


class hittable
{
public:
    //__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const = 0;
    //__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;

	__host__ __device__ virtual ~hittable() = default;

	// pure virtual function 
	// virtual hit method that needs to be implemented for all primitives
	// because each primitive has it own intersection calculation logic
	__device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const = 0;
	__device__ virtual float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const = 0;
	__device__ virtual vector3 random(const vector3& o, curandState* local_rand_state) const = 0;
	__host__ __device__ virtual aabb bounding_box() const = 0;


	__host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableBaseType; }


	

	__host__ __device__ void setName(char* _name)
	{
		m_name = _name;

		//printf("setName1 %s\n", m_name);
	}

	__host__ __device__ void setName(const char* _name)
	{
		m_name = const_cast<char*>(_name);

		//printf("setName2 %s\n", m_name);
	}

	__host__ __device__ char* getName() const
	{
		return m_name;
	}



protected:
	aabb m_bbox;
	uvmapping m_mapping;
	char* m_name;
};
#pragma once

#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"
#include "../lights/light.cuh"
#include "bvh_node.cuh"

class camera;


class scene
{
public:
	__host__ __device__ scene()
	{
	}

	__host__ __device__ void set(hittable** _objects, int list_size);
	__host__ __device__ void set(hittable_list* _objects);
	__host__ __device__ void add(hittable* _object);
	__host__ __device__ void set_camera(camera* _camera);


	__host__ __device__ const hittable_list& get_world();
	__host__ __device__ const hittable_list& get_emissive_objects();
	__host__ __device__ camera* get_camera();

	__host__ __device__ void extract_emissive_objects();
	__device__ void build_optimized_world(curandState& local_rand_state);

private:
	hittable_list m_world;
	camera* m_camera = nullptr;
	hittable_list m_emissive_objects;
};




__host__ __device__ inline void scene::set(hittable** _objects, int list_size)
{
	printf("scene set %i\n", list_size);

	for (int i = 0; i < list_size; i++)
	{
		add(_objects[i]);
	}
}

__host__ __device__ inline void scene::set(hittable_list* _objects)
{
	if (_objects == nullptr || _objects->object_count == 0)
	{
		printf("scene set no data !!!!!\n");
		return;
	}
	
	printf("adding %i objects to scene world\n", _objects->object_count);

	for (int i = 0; i < _objects->object_count; i++)
	{
		add(_objects->objects[i]);
	}
}

__host__ __device__ inline void scene::add(hittable* _object)
{
	if (_object)
	{
		m_world.add(_object);
		printf("scene add to m_world %s\n", _object->getName());
	}
	else
		printf("scene add object nullptr !!!\n");
}

__host__ __device__ inline void scene::set_camera(camera* _camera)
{
	m_camera = _camera;
}

__device__ inline void scene::build_optimized_world(curandState& local_rand_state)
{
	if (m_world.object_count == 0)
	{
		printf("no objs to optmize !!!\n");
		m_world = hittable_list();
		return;
	}
	
	// calculate bounding boxes to speed up ray computing
	hittable* ppp = new bvh_node(m_world.objects, 0, m_world.object_count, &local_rand_state);
	m_world = hittable_list(ppp);

	printf("after build_optimized_world\n");
}

__host__ __device__ inline const hittable_list& scene::get_world()
{
	return m_world;
}

__host__ __device__ inline void scene::extract_emissive_objects()
{
	m_emissive_objects.clear();


	printf("object_count : %i\n", m_world.object_count);


	for (unsigned int i = 0; i < m_world.object_count; i++)
	{
		hittable* nnnnnnn = m_world.objects[i];
		if (nnnnnnn)
		{
			//auto lllllllllll = nnnnnnn->getTypeID();
			//printf("getTypeID %i\n", lllllllllll);

			if (m_world.objects[i]->getTypeID() == HittableTypeID::lightOmniType)
			{
				light* derived = static_cast<light*>(m_world.objects[i]);
				if (derived)
				{
					printf("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ FOUND LIGHT !!!\n");
					m_emissive_objects.add(derived);
				}
			}
		}
		else
		{
			printf("NULL OBJ AT INDEX %i\n", i);
		}
	}
}

__host__ __device__ inline const hittable_list& scene::get_emissive_objects()
{
	return m_emissive_objects;
}

__host__ __device__ inline camera* scene::get_camera()
{
	return m_camera;
}
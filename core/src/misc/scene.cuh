#pragma once

#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"

class camera;


class scene
{
public:
	__host__ __device__ scene()
	{
	}

	__host__ __device__ void set(hittable** _objects, int list_size);
	__host__ __device__ void add(hittable* _object);
	__host__ __device__ void set_camera(camera* _camera);

	__host__ __device__ const hittable_list& get_world();
	__host__ __device__ const hittable_list& get_emissive_objects();
	__host__ __device__ camera* get_camera();

	__host__ __device__ void extract_emissive_objects();
	__host__ __device__ void build_optimized_world();

private:
	hittable_list m_world;
	camera* m_camera = nullptr;
	hittable_list m_emissive_objects;
};

__host__ __device__ void scene::set(hittable** _objects, int list_size)
{
	for (int i = 0; i < list_size; i++)
	{
		add(_objects[i]);
	}
}

__host__ __device__ void scene::add(hittable* _object)
{
	m_world.add(_object); // std::move ????????
}

__host__ __device__ void scene::set_camera(camera* _camera)
{
	m_camera = _camera;
}

__host__ __device__ void scene::build_optimized_world()
{
	// TO FIX !!!!!!!!
	//m_world = hittable_list(new bvh_node(m_world.objects, 0, m_world.object_count));
}

__host__ __device__ const hittable_list& scene::get_world()
{
	return m_world;
}

__host__ __device__ void scene::extract_emissive_objects()
{
	// TO FIX !!!!!!!!!!!!
	/*m_emissive_objects.clear();



	for (unsigned int i = 0; i < m_world.object_count; i++)
	{
		if (m_world.objects[i]->getTypeID() == HittableTypeID::lightType)
		{
			light* derived = static_cast<light*>(m_world.objects[i]);
			if (derived)
			{
				m_emissive_objects.add(derived);
			}
		}
	}*/
}

__host__ __device__ const hittable_list& scene::get_emissive_objects()
{
	return m_emissive_objects;
}

__host__ __device__ camera* scene::get_camera()
{
	return m_camera;
}
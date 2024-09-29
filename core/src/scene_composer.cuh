#pragma once



#include "textures/texture.cuh"
#include "materials/material.cuh"
#include "primitives/hittable_list.cuh"

#include "device_texture.cuh"
#include "device_material.cuh"
#include "device_group.cuh"

#include <thrust/device_vector.h>
#include <thrust/find.h>

class scene_composer
{
public:
	__device__ scene_composer()
	{

	}

	__device__ ~scene_composer() = default;


    __device__ void addObject(hittable* obj);


    __device__ void addSphere(spherePrimitiveConfig sphereCfg);


private:
	thrust::device_vector<device_texture*> m_textures{};
	thrust::device_vector<device_material*> m_materials{};
	thrust::device_vector<device_group*> m_groups{};
	hittable_list m_objects{};

	material* fetchMaterial(const char* name);
	texture* fetchTexture(const char* name);

};

__device__ material* scene_composer::fetchMaterial(const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        for (thrust::device_vector<device_material*>::iterator iter = this->m_materials.begin(); iter != this->m_materials.end(); iter++) 
        {
            auto zzz = (static_cast<device_material*>(*iter));

            if (zzz && zzz->name == name)
            {
                // if key is found
                return zzz->value;
            }
            else
            {
                // if key is not found
                printf("[WARN] Material %s not found !\n", name);
                return nullptr;
            }
        }
    }

    return nullptr;
}

__device__ texture* scene_composer::fetchTexture(const char* name)
{
    if (name != nullptr && name[0] != '\0')
    {
        for (thrust::device_vector<device_texture*>::iterator iter = this->m_textures.begin(); iter != this->m_textures.end(); iter++)
        {
            auto zzz = (static_cast<device_texture*>(*iter));

            if (zzz && zzz->name == name)
            {
                // if key is found
                return zzz->value;
            }
            else
            {
                // if key is not found
                printf("[WARN] Texture %s not found !\n", name);
                return nullptr;
            }
        }
    }

    return nullptr;
}

__device__ void scene_composer::addObject(hittable* obj)
{
  this->m_objects.add(obj);
}


__device__ void scene_composer::addSphere(spherePrimitiveConfig sphereCfg)
{
    //auto sphere = scene_factory::createSphere(name, pos, radius, fetchMaterial(materialName), uv);

    //if (groupName != nullptr && groupName[0] != '\0')
    //{
    //    auto it = this->m_groups.find(groupName);
    //    if (it != this->m_groups.end())
    //    {
    //        // add to existing group is found
    //        hittable_list* grp = it->second;
    //        if (grp) { grp->add(sphere); }
    //    }
    //    else
    //    {
    //        // create group if not found
    //        this->m_groups.emplace(groupName, new hittable_list(sphere));
    //    }
    //}
    //else
    //{
    //    this->m_objects.add(sphere);
    //}
}

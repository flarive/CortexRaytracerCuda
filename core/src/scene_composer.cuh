//#pragma once
//
//
//
//#include "textures/texture.cuh"
//#include "materials/material.cuh"
//#include "primitives/hittable_list.cuh"
//
//#include "device_texture.cuh"
//#include "device_material.cuh"
//#include "device_group.cuh"
//
//#include "scene_factory.cuh"
//
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/find.h>
//
//class scene_composer
//{
//public:
//    __host__ scene_composer()
//	{
//        m_materials = thrust::device_vector<device_material*>(10);
//    }
//
//    __host__ __device__ ~scene_composer() = default;
//
//
//
//    __device__ void addSphere(spherePrimitiveConfig cfg);
//    __device__ void addPlane(planePrimitiveConfig cfg);
//    __device__ void addQuad(quadPrimitiveConfig cfg);
//
//    //__device__ void addLambertianMaterial(lambertianMaterialConfig cfg);
//
//
//    __device__ hittable_list getObjects()
//    {
//        return m_objects;
//    }
//
//
//private:
//	thrust::device_vector<device_texture*> m_textures{};
//    thrust::device_vector<device_material*> m_materials{};
//	thrust::device_vector<device_group*> m_groups{};
//	hittable_list m_objects{};
//
//    
//
//    __device__ material* fetchMaterial(const char* name);
//    __device__ texture* fetchTexture(const char* name);
//
//    __device__ void addObject(hittable* obj);
//};
//
//__device__ material* scene_composer::fetchMaterial(const char* name)
//{
//    if (name != nullptr && name[0] != '\0')
//    {
//        for (thrust::device_vector<device_material*>::iterator iter = this->m_materials.begin(); iter != this->m_materials.end(); iter++)
//        {
//            auto zzz = (static_cast<device_material*>(*iter));
//
//            if (zzz && zzz->name == name)
//            {
//                // if key is found
//                return zzz->value;
//            }
//            else
//            {
//                // if key is not found
//                printf("[WARN] Material %s not found !\n", name);
//                return nullptr;
//            }
//        }
//    }
//
//    return nullptr;
//}
//
//__device__ texture* scene_composer::fetchTexture(const char* name)
//{
//    if (name != nullptr && name[0] != '\0')
//    {
//        for (thrust::device_vector<device_texture*>::iterator iter = this->m_textures.begin(); iter != this->m_textures.end(); iter++)
//        {
//            auto zzz = (static_cast<device_texture*>(*iter));
//
//            if (zzz && zzz->name == name)
//            {
//                // if key is found
//                return zzz->value;
//            }
//            else
//            {
//                // if key is not found
//                printf("[WARN] Texture %s not found !\n", name);
//                return nullptr;
//            }
//        }
//    }
//
//    return nullptr;
//}
//
//__device__ void scene_composer::addObject(hittable* obj)
//{
//    m_objects.add(obj);
//}
//
//
//__device__ void scene_composer::addSphere(spherePrimitiveConfig cfg)
//{
//    auto sphere = scene_factory::createSphere(cfg.name, cfg.position, cfg.radius, fetchMaterial(cfg.materialName), cfg.mapping);
//
//    if (cfg.groupName != nullptr && cfg.groupName[0] != '\0')
//    {
//        //auto it = this->m_groups.find(groupName);
//        //if (it != this->m_groups.end())
//        //{
//        //    // add to existing group is found
//        //    hittable_list* grp = it->second;
//        //    if (grp) { grp->add(sphere); }
//        //}
//        //else
//        //{
//        //    // create group if not found
//        //    this->m_groups.emplace(groupName, new hittable_list(sphere));
//        //}
//    }
//    else
//    {
//        m_objects.add(sphere);
//    }
//}
//
//
//__device__ void scene_composer::addPlane(planePrimitiveConfig cfg)
//{
//    auto plane = scene_factory::createPlane(cfg.name, cfg.point1, cfg.point2, fetchMaterial(cfg.materialName), cfg.mapping);
//    
//    if (cfg.groupName != nullptr && cfg.groupName[0] != '\0')
//    {
//    	//auto it = this->m_groups.find(groupName);
//    	//if (it != this->m_groups.end())
//    	//{
//    	//	// add to existing group is found
//        //       hittable_list* grp = it->second;
//    	//	if (grp) { grp->add(plane); }
//    	//}
//    	//else
//    	//{
//    	//	// create group if not found
//    	//	this->m_groups.emplace(groupName, new hittable_list(plane));
//    	//}
//    }
//    else
//    {
//    	m_objects.add(plane);
//    }
//}
//
//
//__device__ void scene_composer::addQuad(quadPrimitiveConfig cfg)
//{
//    auto quad = scene_factory::createQuad(cfg.name, cfg.position, cfg.u, cfg.v, fetchMaterial(cfg.materialName), cfg.mapping);
//       
//    if (cfg.groupName != nullptr && cfg.groupName[0] != '\0')
//    {
//        //auto it = this->m_groups.find(groupName);
//        //if (it != this->m_groups.end())
//        //{
//        //    // add to existing group is found
//        //        hittable_list* grp = it->second;
//        //    if (grp) { grp->add(quad); }
//        //}
//        //else
//        //{
//        //    // create group if not found
//        //    this->m_groups.emplace(groupName, new hittable_list(quad));
//        //}
//    }
//    else
//    {
//        m_objects.add(quad);
//    }
//}
//
//
////__device__ void scene_composer::addLambertianMaterial(lambertianMaterialConfig cfg)
////{
////    device_material* mat = new device_material(cfg.name, new lambertian(cfg.rgb));
////
////
////}
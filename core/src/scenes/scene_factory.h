//#pragma once
//
//#include "../primitives/hittable.cuh"
//#include "../materials/material.cuh"
//#include "../misc/vector3.cuh"
//
//class scene_factory
//{
//public:
//	scene_factory() = delete;
//
//	static hittable* createBox(const char* name, const point3& p0, const point3& p1, material* material, const uvmapping& uv);
//
//	static hittable* createCylinder(const char* name, const point3& center, float radius, float height, material* material, const uvmapping& uv);
//
//	static hittable* createSphere(const char* name, const point3& center, float radius, material* material, const uvmapping& uv);
//
//	static hittable* createCone(const char* name, const point3& center, float height, float radius, material* material, const uvmapping& uv);
//
//	static hittable* createDisk(const char* name, const point3& center, float height, float radius, material* material, const uvmapping& uv);
//
//	static hittable* createTorus(const char* name, const point3& center, float major_radius, float minor_radius, material* material, const uvmapping& uv);
//
//	static hittable* createQuad(const char* name, const point3& position, const vector3 u, const vector3 v, material* material, const uvmapping& uv);
//
//	static hittable* createPlane(const char* name, const point3& p0, point3 p1, material* material, const uvmapping& uv);
//
//	static hittable* createVolume(const char* name, hittable* boundary, float density, texture* texture);
//
//	static hittable* createVolume(const char* name, hittable* boundary, float density, const color& rgb);
//
//	static hittable* createMesh(const char* name, const point3& center, const char* filepath, material* material, const bool use_mtl, const bool use_smoothing);
//
//	static hittable* createDirectionalLight(const char* name, const point3& pos, const vector3& u, const vector3& v, float intensity, color rgb, bool invisible);
//
//	static hittable* createOmniDirectionalLight(const char* name, const point3& pos, float radius, float intensity, color rgb, bool invisible);
//
//	static hittable* createSpotLight(const char* name, const point3& pos, const vector3& dir, float cutoff, float falloff, float intensity, float radius, color rgb, bool invisible);
//};
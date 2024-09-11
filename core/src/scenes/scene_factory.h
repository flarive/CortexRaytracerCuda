//#pragma once
//
//#include "../primitives/hittable.cuh"
//#include "../materials/material.cuh"
//#include "../misc/vector3.cuh"
//
//
//class scene_factory
//{
//public:
//	scene_factory() = delete;
//
//	static std::shared_ptr<hittable> createBox(const std::string name, const point3& p0, const point3& p1, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createCylinder(const std::string name, const point3& center, double radius, double height, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createSphere(const std::string name, const point3& center, double radius, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createCone(const std::string name, const point3& center, double height, double radius, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createDisk(const std::string name, const point3& center, double height, double radius, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createTorus(const std::string name, const point3& center, double major_radius, double minor_radius, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createQuad(const std::string name,	const point3& position, const vector3 u, const vector3 v, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createPlane(const std::string name, const point3& p0, point3 p1, const std::shared_ptr<material>& material, const uvmapping& uv);
//
//	static std::shared_ptr<hittable> createVolume(const std::string name, const std::shared_ptr<hittable>& boundary, double density, std::shared_ptr<texture> texture);
//
//	static std::shared_ptr<hittable> createVolume(const std::string name, const std::shared_ptr<hittable>& boundary, double density, const color& rgb);
//
//	static std::shared_ptr<hittable> createMesh(const std::string name, const point3& center, const std::string filepath, const std::shared_ptr<material>& material, const bool use_mtl, const bool use_smoothing);
//
//	static std::shared_ptr<hittable> createDirectionalLight(std::string name, const point3& pos, const vector3& u, const vector3& v, double intensity, color rgb, bool invisible);
//
//	static std::shared_ptr<hittable> createOmniDirectionalLight(std::string name, const point3& pos, double radius, double intensity, color rgb, bool invisible);
//
//	static std::shared_ptr<hittable> createSpotLight(std::string name, const point3& pos, const vector3& dir, double cutoff, double falloff, double intensity, double radius, color rgb, bool invisible);
//};
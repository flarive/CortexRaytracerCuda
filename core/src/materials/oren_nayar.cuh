#pragma once

#include "material.cuh"
#include "../primitives/hittable.cuh"
#include "../textures/texture.cuh"
#include "../primitives/hittable_list.cuh"
#include "../lights/light.cuh"

#include "../pdf/cosine_pdf.cuh"
#include "../textures/solid_color_texture.cuh"

/// <summary>
/// Oren Nayar material
/// For natural surfaces, such as concrete, plaster, sand, etc.
/// </summary>
class oren_nayar : public material
{
public:
	//__host__ __device__ oren_nayar(color _color);
	//__host__ __device__ oren_nayar(color _color, float _albedo_temp, float _roughness);

	//__host__ __device__ oren_nayar(texture* _albedo);
	//__host__ __device__ oren_nayar(texture* _albedo, float _albedo_temp, float _roughness);



	__host__ __device__ oren_nayar(color _color) :
		material(new solid_color_texture(_color)), m_albedo_temp(0.30f), m_roughness(0.5f)
	{
	}

	__host__ __device__ oren_nayar(texture* _albedo) :
		material(_albedo), m_albedo_temp(0.30f), m_roughness(0.5f)
	{
	}

	__host__ __device__ oren_nayar(color _color, float _albedo_temp, float _roughness)
		: material(new solid_color_texture(_color)), m_albedo_temp(_albedo_temp), m_roughness(_roughness)
	{
	}

	__host__ __device__ oren_nayar(texture* _albedo, float _albedo_temp, float _roughness) :
		material(_albedo), m_albedo_temp(_albedo_temp), m_roughness(_roughness)
	{
	}



	__device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const override;

	__host__ __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override;

	__host__ __device__ MaterialTypeID getTypeID() const override { return MaterialTypeID::materialOrenNayarType; }



private:
	float m_roughness = 0.0f;
	float m_refractiveIndex = 0.0f;
	float m_reflectivity = 0.0f;
	float m_transparency = 0.0f;
	float m_emissivity = 0.0f;
	float m_specularity = 0.0f;
	float m_specularExponent = 0.0f;

	// ?????
	float m_albedo_temp = 0.0f;
};



// https://github.com/Friduric/ray-tracer/blob/master/src/Rendering/Materials/OrenNayarMaterial.cpp
//bool oren_nayar::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec) const
//{
//	vector3 inDirection = -rec.normal;
//	vector3 outDirection = -r_in.direction();
//	vector3 normal = rec.normal;
//
//	// just take the first light for the moment
//	std::shared_ptr<light> mylight = std::dynamic_pointer_cast<light>(lights.objects[0]);
//	if (mylight == nullptr)
//	{
//		// no light
//		return false;
//	}
//
//	// Combine the surface color with the light's color/intensity
//	color incomingIntensity = mylight->getColor() * mylight->getIntensity();
//
//
//
//
//	assert(glm::length(inDirection) < 1.0f + 10.0f * FLT_EPSILON && glm::length(inDirection) > 1.0f - 10.0f * FLT_EPSILON);
//	assert(glm::length(outDirection) < 1.0f + 10.0f * FLT_EPSILON && glm::length(outDirection) > 1.0f - 10.0f * FLT_EPSILON);
//	// See https://en.wikipedia.org/wiki/Oren-Nayar_reflectance_model for more information.
//		
//	const float indot = glm::dot(-inDirection, normal);
//	if (indot < FLT_EPSILON) {
//		return false;
//	}
//		
//	const float sigma2 = roughness * roughness;
//		
//	const float A = 1 - 0.5f * sigma2 / (sigma2 + 0.57f);
//	const float B = 0.45f * sigma2 / (sigma2 + 0.09f);
//		
//	const float alphaInclination = acos(indot);
//	const float betaInclination = acos(glm::dot(outDirection, normal));
//		
//	const float alpha = std::max(alphaInclination, betaInclination);
//	const float beta = std::min(alphaInclination, betaInclination);
//		
//	const float C = glm::max<float>(0.0f, glm::dot(-inDirection, outDirection));
//	const float D = (C > 0.0f) ? B * C * sin(alpha) * tan(beta) : 0.0f;
//		
//	color final_color = glm::max(0.0f, indot) * glm::min<float>((A + D), 1.0f) * (incomingIntensity * surfaceColor);
//
//
//	srec.attenuation = final_color;
//	//srec.pdf_ptr = nullptr;
//	//srec.skip_pdf = true;
//	//srec.skip_pdf_ray = ray(rec.hit_point, random_in_unit_sphere(), r_in.time());
//	srec.pdf_ptr = std::make_shared<sphere_pdf>();
//	srec.skip_pdf = false;
//
//	return true;
//}

/// <summary>
/// ChatGPT implem
/// </summary>
/// <param name="r_in"></param>
/// <param name="lights"></param>
/// <param name="rec"></param>
/// <param name="srec"></param>
/// <returns></returns>
__device__ inline bool oren_nayar::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const
{
	vector3 scatterDirection = rec.normal + random_on_hemisphere(rng, rec.normal);
	color mycolor = m_diffuse_texture->value(rec.u, rec.v, rec.hit_point);

	// just take the first light for the moment
	//std::shared_ptr<light> mylight = std::dynamic_pointer_cast<light>(lights.objects[0]);
	//if (mylight == nullptr)
	//{
	//	// no light
	//	return false;
	//}

	light* mylight;

	if (lights.objects[0]->getTypeID() == HittableTypeID::lightType
		|| lights.objects[0]->getTypeID() == HittableTypeID::lightDirectionalType
		|| lights.objects[0]->getTypeID() == HittableTypeID::lightOmniType
		|| lights.objects[0]->getTypeID() == HittableTypeID::lightSpotType)
	{
		mylight = static_cast<light*>(lights.objects[0]);
	}
	else
	{
		// no light
		return false;
	}


	// Combine the surface color with the light's color/intensity
	color incomingIntensity = mylight->getColor() * mylight->getIntensity();


	srec.pdf_ptr = new sphere_pdf();
	srec.skip_pdf = false;

	srec.attenuation = incomingIntensity * mycolor * m_albedo_temp * (1.0f / M_PI); // Lambertian reflection

	float cosThetaI = glm::dot(-r_in.direction(), rec.normal);
	float cosThetaO = glm::dot(srec.skip_pdf_ray.direction(), rec.normal);

	// Oren-Nayar diffuse reflection model
	float sigma = m_roughness * m_roughness;
	float A = 1.0f - 0.5f * (sigma / (sigma + 0.33f));
	float B = 0.45f * (sigma / (sigma + 0.09f));
	float maxCos = fmax(0.0f, cosThetaI);
	float maxCosPrime = fmax(0.0f, cosThetaO);
	float sinThetaI = sqrt(1.0f - maxCos * maxCos);
	float sinThetaO = sqrt(1.0f - maxCosPrime * maxCosPrime);
	float alpha = fmax(sinThetaI, sinThetaO) / fmin(sinThetaI, sinThetaO);
	float beta = fmin(sinThetaI, sinThetaO) / fmax(sinThetaI, sinThetaO);
	float diffuseReflectance = A + B * alpha * beta;
	srec.attenuation *= diffuseReflectance;

	return true;
}



__host__ __device__ inline float oren_nayar::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
	auto cos_theta = glm::dot(rec.normal, unit_vector(scattered.direction()));
	return cos_theta < 0 ? 0.0f : cos_theta / M_PI;
}






/// <summary>
/// TO DO !!! This implem seems better !!!!!!!!!!!
/// https://blog.csdn.net/lr_shadow/article/details/124938722
/// </summary>
//class oren_nayar2 : public material
//{
//public:
//	oren_nayar2(color c, double sigma) : albedo(std::make_shared<solid_color_texture>(c))
//	{
//		double sigma2 = sigma * sigma;
//		a = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
//		b = 0.45 * sigma2 / (sigma2 + 0.09);
//	}
//
//	oren_nayar2(std::shared_ptr<texture> albedo_, double sigma) : albedo(albedo_)
//	{
//
//		double sigma2 = sigma * sigma;
//		a = 1.0 - (sigma2 / (2.0 * (sigma2 + 0.33)));
//		b = 0.45 * sigma2 / (sigma2 + 0.09);
//	}
//
//
//	bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec) const
//	{
//		srec.skip_pdf = false;
//		srec.attenuation = albedo->value(rec.u, rec.v, rec.hit_point);
//		srec.pdf_ptr = std::make_shared<cosine_pdf>(rec.normal);
//		return true;
//	}
//
//	// eval brdf
//	double scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
//	{
//		vector3 wi = normalize(r_in.direction());
//		vector3 wo = normalize(scattered.direction());
//
//		double cosine = dot(rec.normal, wo);
//		if (cosine < 0)
//			cosine = 0;
//
//		double sinThetaI = SinTheta(wi);
//		double sinThetaO = SinTheta(wo);
//
//		double maxCos = 0;
//
//		if (sinThetaI > 1e-4 && sinThetaO > 1e-4)
//		{
//			double sinPhiI = SinPhi(wi);
//			double cosPhil = CosPhi(wi);
//
//			double sinPhio = SinPhi(wo);
//			double cosphio = CosPhi(wo);
//			double dcos = cosPhiI * cosPhio + sinPhil * sinPhio;
//			maxCos = std::max(0.0, dcos);
//		}
//
//		double sinAlpha, tanBeta;
//		if (AbsCosTheta(wi) > AbsCosTheta(wo))
//		{
//			sinAlpha = sinThetaO;
//			tanBeta = sinThetaI / AbsCosTheta(wi);
//		}
//		else
//		{
//			sinAlpha = sinThetaI;
//			tanBeta = sinThetaO / AbsCosTheta(wo);
//		}
//		return ((a + b * maxCos * sinAlpha * tanBeta) * INV_PI * cosine);
//	}
//public:
//	shared_ptr<texture> albedo;
//	double a, b;
//};
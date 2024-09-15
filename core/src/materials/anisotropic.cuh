#pragma once

#include "material.cuh"
#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"
#include "../textures/texture.cuh"

#include "../pdf/cosine_pdf.cuh"
#include "../pdf/anisotropic_phong_pdf.cuh"
#include "../textures/solid_color_texture.cuh"



/// <summary>
/// Anisotropic material
/// Anisotropic materials show different properties in different directions.
/// Wood, composite materials, all crystals (except cubic crystal) are examples of anisotropic materials.
/// Is a material (shader) that represents surfaces with grooves, such as a CD, feathers, or fabrics like velvet or satin
/// </summary>
class anisotropic : public material
{
public:
    //__host__ __device__ anisotropic(double Nu, double Nv, texture* diffuseTexture, texture* specularTexture, texture* exponentTexture);


    __host__ __device__ anisotropic(float Nu, float Nv, texture* diffuseTexture, texture* specularTexture, texture* exponentTexture)
        : m_diffuse(diffuseTexture), m_specular(specularTexture), m_exponent(exponentTexture), m_nu(Nu), m_nv(Nv)
    {
    }



    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const override;

    __host__ __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override;

	__host__ __device__ MaterialTypeID getTypeID() const override { return MaterialTypeID::materialAnisotropicType; }


private:
    const float epsilon = 1E-5;

    texture* m_diffuse;
    texture* m_specular;
    texture* m_exponent;

    // exponents - for nu=nv it's similar with Phong
    // The Phong exponent defines the "tightness" of the highlight.
    // A higher exponent results in a smaller, tighter highlight while a lower exponent results in a broader flatter one.
    float m_nu = 0.0f;
    float m_nv = 0.0f;


    __host__ __device__ color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p) const;
};




__device__ bool anisotropic::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const
{
	srec.skip_pdf = true;
	srec.attenuation = srec.diffuseColor = m_diffuse->value(rec.u, rec.v, rec.hit_point);

	if (m_specular)
		srec.specularColor = m_specular->value(rec.u, rec.v, rec.hit_point);

	if (m_exponent)
	{
		const color c = m_exponent->value(rec.u, rec.v, rec.hit_point);
		const double e = (c.r() + c.g() + c.b()) / 3.;

		if (e > 0)
			srec.pdf_ptr = new anisotropic_phong_pdf(r_in.direction(), rec.normal, e * m_nu, e * m_nv);
		else
			srec.pdf_ptr = new anisotropic_phong_pdf(r_in.direction(), rec.normal, m_nu, m_nv);
	}
	else
	{
		srec.pdf_ptr = new anisotropic_phong_pdf(r_in.direction(), rec.normal, m_nu, m_nv);
	}


	vector3 dir(srec.pdf_ptr->generate(srec, rng));


	while (vector_multiply_to_double(dir, rec.normal) < 0)
	{
		dir = srec.pdf_ptr->generate(srec, rng);
	}
	srec.skip_pdf_ray = ray(rec.hit_point + epsilon * rec.normal, dir);

	return true;
}

__host__ __device__ color anisotropic::emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p) const
{
	return color::black();
}


__host__ __device__ float anisotropic::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
	const float cosine = vector_multiply_to_double(rec.normal, scattered.direction());

	if (cosine < 0) return 0.0f;

	return cosine * M_1_PI;
}


/// Github copilot
//#include "material.h"
//
//class AnisotropicMaterial : public Material {
//public:
//	// Constructeur
//	AnisotropicMaterial(const Vec3& albedo, float fuzziness)
//		: albedo_(albedo), fuzziness_(fuzziness) {}
//
//	// Fonction de dispersion
//	bool scatter(const Ray& ray_in, const HitRecord& rec, ScatterRecord& srec) const override {
//		// Calcul de la direction réfléchie
//		Vec3 reflected = reflect(unit_vector(ray_in.direction()), rec.normal);
//		srec.specular_ray = Ray(rec.p, reflected + fuzziness_ * random_in_unit_sphere());
//		srec.attenuation = albedo_;
//		srec.is_specular = true;
//		srec.pdf_ptr = nullptr;
//		return true;
//	}
//
//private:
//	Vec3 albedo_;
//	float fuzziness_;
//};
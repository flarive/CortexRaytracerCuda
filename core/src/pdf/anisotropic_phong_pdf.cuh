#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../misc/gpu_randomizer.cuh"
#include "../misc/scatter_record.cuh"

class anisotropic_phong_pdf : public pdf
{
public:
	__device__ anisotropic_phong_pdf(const vector3& inc, const vector3& norm, double Nu, double Nv)
		: m_incident(inc), m_onb(norm, inc), m_nu(Nu), m_nv(Nv)
	{
		const double nu1 = m_nu + 1.;
		const double nv1 = m_nv + 1.;
		m_prefactor1 = sqrt(nu1 / nv1);
		m_prefactor2 = sqrt(nu1 * nv1) / (2.0f * M_PI);
	}

	__device__ ~anisotropic_phong_pdf() = default;

	__device__ float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const override;
	__device__ vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) override;

	__host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfAnisotropicPhong; }

private:
	__host__ __device__ inline static double Schlick(const double val, float cosine)
	{
		return val + (1.0f - val) * pow(1.0f - cosine, 5.0f);
	}

	__host__ __device__ inline vector3 GetSpecularReflected(const vector3& h, float kh) const
	{
		return m_incident + 2.0f * kh * h;
	}

	__host__ __device__ inline double GetSpecularPDH(const vector3& h, float kh, float cos2, float sin2) const
	{
		return GetHPDH(h, cos2, sin2) / (4.0f * kh);
	}

	__host__ __device__ inline double GetHPDH(const vector3& h, float cos2, float sin2) const
	{
		auto kkk = m_onb.Normal();
		double nh = h.x * kkk.x + h.y * kkk.y + h.z * kkk.z;

		return m_prefactor2 * glm::pow(nh, m_nu * cos2 + m_nv * sin2);
	}

	__host__ __device__ static void DealWithQuadrants(float& xi, float& phase, bool& flip);

	vector3 m_incident{};
	onb m_onb;

	float m_nu = 0.0f;
	float m_nv = 0.0f;

	float m_prefactor1 = 0.0f;
	float m_prefactor2 = 0.0f;
};



__device__ inline float anisotropic_phong_pdf::value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const
{
	const float cosine = vector_multiply_to_double(direction, m_onb.Normal());
	if (cosine < 0) return 0;

	return cosine * M_1_PI;
}

__device__ inline vector3 anisotropic_phong_pdf::generate(scatter_record& rec, thrust::default_random_engine& rng)
{
	float phase;
	bool flip;

	float xi = get_real(rng);
	DealWithQuadrants(xi, phase, flip);

	float phi = atan(m_prefactor1 * tan(M_PI_2 * xi));
	if (flip)
		phi = phase - phi;
	else
		phi += phase;

	const float c = cos(phi);
	const float s = sin(phi);
	const float c2 = c * c;
	const float s2 = 1. - c2;

	xi = get_real(rng);
	DealWithQuadrants(xi, phase, flip);

	float theta = acos(pow(1.0f - xi, 1.0f / (m_nu * c2 + m_nv * s2 + 1.0f)));
	if (flip)
		theta = phase - theta;
	else
		theta += phase;

	const float st = sin(theta);
	const float ct = cos(theta);

	const float cos2 = ct * ct;
	const float sin2 = st * st;

	const vector3 h = m_onb.LocalToGlobal(vector3(st * c, st * s, ct));

	float diffuseProbability;
	float kh = 0.; // avoid complains about not being initialized

	if (vector_multiply_to_double(h, m_onb.Normal()) < 0)
		diffuseProbability = 1.;
	else
	{
		kh = vector_multiply_to_double(-m_incident, h);
		const float specularProbability = GetSpecularPDH(h, kh, cos2, sin2);
		const float weight = 1. + specularProbability;

		diffuseProbability = 1. / weight;
	}

	if (get_real(rng) < diffuseProbability)
	{
		rec.attenuation = rec.diffuseColor;
		return m_onb.LocalToGlobal(random_cosine_direction(rng));
	}

	// I don't like the white specular color that's typical in obj files, mix it with the diffuse color
	rec.attenuation = 0.8 * rec.specularColor + 0.2f * rec.diffuseColor;

	return GetSpecularReflected(h, kh);
}

__host__ __device__ inline void anisotropic_phong_pdf::DealWithQuadrants(float& xi, float& phase, bool& flip)
{
	phase = 0;
	flip = false;

	if (xi < 0.25f)
	{
		xi *= 4;
	}
	else if (xi < 0.5f)
	{
		xi = 1.0f - 4.0f * (0.5f - xi);
		phase = M_PI;
		flip = true;
	}
	else if (xi < 0.75)
	{
		xi = 1.0f - 4.0f * (0.75f - xi);
		phase = M_PI;
	}
	else
	{
		xi = 1.0f - 4.0f * (1.0f - xi);
		phase = 2.0f * M_PI;
		flip = true;
	}
}
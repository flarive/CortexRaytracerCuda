#pragma once

//#include "ray.cuh"
#include "../misc/vector3.cuh"

#include <cmath>

/// <summary>
/// Ortho Normal Basis
/// An orthonormal basis (ONB) is a collection of three mutually orthogonal unit vectors
/// </summary>
class onb
{
public:
	__device__ onb()
	{
	};

	// this will have passed the normal to the surface
	__device__ onb(const vector3& v)
	{
		BuildFrom(v);
	};

	// the normal and the incident ray vector
	__device__ onb(const vector3& n, const vector3& i)
	{
		BuildFrom(n, i);
	};

	// normal, tangent, bitangent - in case it's also used for the TBN matrix
	__device__ onb(const vector3& n, const vector3& t, const vector3& b)
	{
		BuildFrom(n, t, b);
	};

	__device__ ~onb() = default;

	__device__ vector3 operator[](int i) const
	{
		return m_axis[i];
	}

	__host__ __device__ vector3 u() const
	{
		return m_axis[0];
	}

	__host__ __device__ vector3 v() const
	{
		return m_axis[1];
	}

	__host__ __device__ vector3 w() const
	{
		return m_axis[2];
	}

	__device__ vector3 local(float a, float b, float c) const;


	__device__ vector3 local(const vector3& a) const;


	__device__ void build_from_w(const vector3& n);

	__host__ __device__ inline vector3 Normal() const
	{
		return w();
	}

	__device__ inline vector3 LocalToGlobal(float X, float Y, float Z) const;

	__device__ inline vector3 LocalToGlobal(const vector3& vect) const;

	__device__ inline vector3 GlobalToLocal(const vector3& vect) const;


	__device__ inline void BuildFrom(const vector3& v);

	__device__ inline void BuildFrom(const vector3& n, const vector3& i);

	__device__ inline void BuildFrom(const vector3& n, const vector3& t, const vector3& b);



private:
	vector3 m_axis[3]{};
};


__device__ inline vector3 onb::LocalToGlobal(float X, float Y, float Z) const
{
	return X * m_axis[0] + Y * m_axis[1] + Z * m_axis[2];
}

__device__ inline vector3 onb::LocalToGlobal(const vector3& vect) const
{
	return vect.x * m_axis[0] + vect.y * m_axis[1] + vect.z * m_axis[2];
}

__device__ inline vector3 onb::GlobalToLocal(const vector3& vect) const
{
	return vector3(
		m_axis[0].x * vect.x + m_axis[1].x * vect.y + m_axis[2].x * vect.z,
		m_axis[0].y * vect.x + m_axis[1].y * vect.y + m_axis[2].y * vect.z,
		m_axis[0].z * vect.x + m_axis[1].z * vect.y + m_axis[2].z * vect.z
	);
}


__device__ inline vector3 onb::local(float a, float b, float c) const
{
	return a * u() + b * v() + c * w();
}

__device__ inline vector3 onb::local(const vector3& a) const
{
	return a.x * u() + a.y * v() + a.z * w();
}

__device__ inline void onb::build_from_w(const vector3& n)
{
	m_axis[2] = unit_vector(n);
	vector3 a = (fabs(w().x) > 0.9f) ? vector3(0, 1, 0) : vector3(1, 0, 0);
	m_axis[1] = unit_vector(glm::cross(w(), a));
	m_axis[0] = glm::cross(w(), v());
}

__device__ inline void onb::BuildFrom(const vector3& v)
{
	m_axis[2] = v;

	if (abs(m_axis[2].x) > 0.9f) m_axis[1] = vector3(0, 1, 0);
	else m_axis[1] = vector3(1, 0, 0);

	m_axis[1] = glm::normalize(vector_modulo_operator(m_axis[1], m_axis[2]));
	m_axis[0] = vector_modulo_operator(m_axis[1], m_axis[2]); // x = y % z
}

__device__ inline void onb::BuildFrom(const vector3& n, const vector3& i)
{
	const float cosine = glm::abs(vector_multiply_to_double(n, i));

	if (cosine > 0.99999999f)
	{
		BuildFrom(n);
	}
	else
	{
		m_axis[2] = n; // z
		m_axis[0] = glm::normalize(vector_modulo_operator(i, n)); // x
		m_axis[1] = vector_modulo_operator(m_axis[2], m_axis[0]); // x = z % x
	}
}

__device__ inline void onb::BuildFrom(const vector3& n, const vector3& t, const vector3& b)
{
	m_axis[2] = n; // z
	m_axis[0] = t;
	m_axis[1] = b;
}
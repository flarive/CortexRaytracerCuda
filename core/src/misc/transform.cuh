#pragma once

#include "vector3.cuh"

namespace rt
{
	class transform
	{
	public:
		__host__ __device__ void setTranslate(const vector3& v);
		__host__ __device__ void setRotate(const vector3& v);
		__host__ __device__ void setScale(const vector3& v);

		__host__ __device__ vector3 getTranslate() const;
		__host__ __device__ vector3 getRotate() const;
		__host__ __device__ vector3 getScale() const;

		__host__ __device__ bool hasTranslate() const;
		__host__ __device__ bool hasRotate() const;
		__host__ __device__ bool hasScale() const;

	private:
		 bool m_hasTranslate = false;
		 bool m_hasRotate = false;
		 bool m_hasScale = false;

		 vector3 m_translate{};
		 vector3 m_rotate{};
		 vector3 m_scale{};
	};
}


__host__ __device__ inline void rt::transform::setTranslate(const vector3& v)
{
	m_translate = v;
	m_hasTranslate = true;
}

__host__ __device__ inline void rt::transform::setRotate(const vector3& v)
{
	m_rotate = v;
	m_hasRotate = true;
}

__host__ __device__ inline void rt::transform::setScale(const vector3& v)
{
	m_scale = v;
	m_hasScale = true;
}

__host__ __device__ inline vector3 rt::transform::getTranslate() const
{
	return m_translate;
}

__host__ __device__ inline vector3 rt::transform::getRotate() const
{
	return m_rotate;
}

__host__ __device__ inline vector3 rt::transform::getScale() const
{
	return m_scale;
}

__host__ __device__ inline bool rt::transform::hasTranslate() const
{
	return m_hasTranslate;
}

__host__ __device__ inline bool rt::transform::hasRotate() const
{
	return m_hasRotate;
}

__host__ __device__ inline bool rt::transform::hasScale() const
{
	return m_hasScale;
}
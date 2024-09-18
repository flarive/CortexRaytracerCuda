#pragma once

#include "pdf.cuh"
#include "../misc/constants.cuh"
#include "../utilities/uvmapping.cuh"
#include "../misc/gpu_randomizer.cuh"

// https://github.com/Drummersbrother/raytracing-in-one-weekend/blob/90b1d3d7ce7f6f9244bcb925c77baed4e9d51705/main.cpp#L26
class image_pdf : public pdf
{
public:
	__device__ image_pdf(image_texture* img);

	__device__ ~image_pdf();

	__device__ float value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const override;
	__device__ vector3 generate(scatter_record& rec, thrust::default_random_engine& rng) override;

	__host__ __device__ virtual pdfTypeID getTypeID() const { return pdfTypeID::pdfImage; }

private:
	image_texture* m_image = nullptr;
	unsigned int m_width = 0, m_height = 0, m_channels = 0;
	float* m_pUDist;
	float* m_pBuffer;
	float* m_pData;

	__device__ float* device_lower_bound(float* first, float* last, float value);
};

__device__ image_pdf::image_pdf(image_texture* img)
	: m_image(img), m_width(img->getWidth()), m_height(img->getHeight()), m_channels(3), m_pData(img->get_data_float())
{
	unsigned int k = 0;
	float angleFrac = M_PI / m_height;
	float theta = static_cast<float>(angleFrac) * 0.5f;
	float sinTheta = 0.f;
	float* pSinTheta = (float*)malloc(sizeof(float) * m_height);
	for (unsigned int i = 0; i < m_height; i++, theta += angleFrac)
	{
		pSinTheta[i] = sin(theta);
	}
	// convert the data into a marginal CDF along the columns
	m_pBuffer = (float*)malloc(m_width * (m_height + 1) * sizeof(float));

	if (m_pBuffer)
	{
		m_pUDist = &m_pBuffer[m_width * m_height];
		if (m_pUDist)
		{
			for (unsigned int i = 0, m = 0; i < m_width; i++, m += m_height)
			{
				float* pVDist = &m_pBuffer[m];
				if (pVDist)
				{
					k = i * m_channels;
					pVDist[0] = 0.2126f * m_pData[k + 0] + 0.7152f * m_pData[k + 1] + 0.0722f * m_pData[k + 2];
					pVDist[0] *= pSinTheta[0];
					for (unsigned int j = 1, k = (m_width + i) * m_channels; j < m_height; j++, k += m_width * m_channels)
					{
						float lum = 0.2126f * m_pData[k + 0] + 0.7152f * m_pData[k + 1] + 0.0722f * m_pData[k + 2];
						lum *= pSinTheta[j];
						pVDist[j] = pVDist[j - 1] + lum;
					}

					if (i == 0)
					{
						m_pUDist[i] = pVDist[m_height - 1];
					}
					else
					{
						m_pUDist[i] = m_pUDist[i - 1] + pVDist[m_height - 1];
					}
				}
			}
		}
	}
}

__device__ inline float image_pdf::value(const vector3& direction, int max_depth, thrust::default_random_engine& rng) const
{
	float _u, _v;
	get_spherical_uv(unit_vector(direction), _u, _v);
	_u = 1.0f - _u;
	unsigned int u = int(_u * float(m_height - 1));
	unsigned int v = int(_v * float(m_width - 1));
	if (u < 0) u = 0;
	if (u >= m_height) u = m_height - 1;

	if (v < 0) v = 0;
	if (v >= m_width) v = m_width - 1;

	float angleFrac = M_PI / float(m_height);
	float invPdfNorm = (2.0f * float(M_PI * M_PI)) / float(m_width * m_height);
	float* pVDist = &m_pBuffer[m_height * u];
	// compute the actual PDF
	float pdfU = (u == 0) ? (m_pUDist[0]) : (m_pUDist[u] - m_pUDist[u - 1]);
	pdfU /= m_pUDist[m_width - 1];
	float pdfV = (v == 0) ? (pVDist[0]) : (pVDist[v] - pVDist[v - 1]);
	pdfV /= pVDist[m_height - 1];
	float theta = angleFrac * 0.5f + angleFrac * u;
	float Pdf = (pdfU * pdfV) * sin(theta) / invPdfNorm;

	return Pdf;
}

__device__ inline vector3 image_pdf::generate(scatter_record& rec, thrust::default_random_engine& rng)
{
	float r1 = get_real(rng);
	float r2 = get_real(rng);

	float maxUVal = m_pUDist[m_width - 1];
	float* pUPos = device_lower_bound(m_pUDist, m_pUDist + m_width, r1 * maxUVal);
	int u = int(pUPos - m_pUDist);
	float* pVDist = &m_pBuffer[m_height * u];
	float* pVPos = device_lower_bound(pVDist, pVDist + m_height, r2 * pVDist[m_height - 1]);
	int v = int(pVPos - pVDist);

	float _u = float(u) / m_height, _v = float(v) / m_width;
	_u = 1.0f - _u;

	return from_spherical_uv(_u, _v);
}

__device__ inline float* image_pdf::device_lower_bound(float* first, float* last, float value)
{
	float* low = first;
	float* high = last;

	while (low < high) {
		float* mid = low + (high - low) / 2;

		if (*mid < value) {
			low = mid + 1;
		}
		else {
			high = mid;
		}
	}

	return low;  // Return pointer to the first element not less than `value`
}

// Destructor implementation
__device__ inline image_pdf::~image_pdf()
{
	printf("Calling image_pdf destructor\n");

	if (m_image) {
		delete m_image;
		m_image = nullptr;
	}

	if (m_pUDist)
	{
		delete m_pUDist;
		m_pUDist = nullptr;
	}

	if (m_pBuffer)
	{
		delete m_pBuffer;
		m_pBuffer = nullptr;
	}

	if (m_pData)
	{
		delete m_pData;
		m_pData = nullptr;
	}
}

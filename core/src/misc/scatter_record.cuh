#pragma once

#include "color.cuh"
#include "ray.cuh"
#include "../pdf/pdf.cuh"

struct scatter_record
{
public:
	color attenuation{};
	pdf* pdf_ptr = nullptr;
	bool skip_pdf = false; // is specular
	ray skip_pdf_ray; // specular_ray

	color diffuseColor{};  // used only by AnisotropicPhong
	color specularColor{}; // used only by AnisotropicPhong

	float alpha_value = 1.0f; // If no alpha texture, return 1.0 (fully opaque)

	__device__ ~scatter_record()
	{
		if (pdf_ptr) {
			/*delete pdf_ptr;
			pdf_ptr = nullptr;*/
		}
	}
};
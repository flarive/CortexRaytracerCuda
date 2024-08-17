#pragma once

#include "misc/scene.h"
#include "misc/color.cuh"
#include "cameras/camera.cuh"
#include "misc/render_parameters.h"
#include "samplers/sampler.cuh"

class renderer
{
public:
	renderer();
	virtual ~renderer() = default;

	virtual void render(scene& _scene, camera& _camera, const render_parameters& _params, sampler* aa_sampler) const = 0;

protected:
	static void preview_line(int j, std::vector<color> i, int spp);
	static bool saveToFile(std::string filepath, std::vector<std::vector<color>> image, int width, int height, int spp);
};


renderer::renderer()
{
}

void renderer::preview_line(int j, std::vector<color> i, int spp)
{
	for (unsigned int n = 0; n < i.size(); n++)
	{
		//color::write_color(std::cout, n, j, i[n], spp);
	}
}

bool renderer::saveToFile(std::string filepath, std::vector<std::vector<color>> image, int width, int height, int spp)
{
	//// save image to disk
	//uint8_t* data = bitmap_image::buildPNG(image, width, height, spp, true);
	//if (data)
	//{
	//	constexpr int CHANNELS = 3;
	//	return bitmap_image::saveAsPNG(filepath, width, height, CHANNELS, data, width * CHANNELS);
	//}

	return false;
}
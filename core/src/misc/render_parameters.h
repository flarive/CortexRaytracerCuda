#pragma once

#include <iostream>
#include <sstream>
#include <string>

class render_parameters
{
public:

	bool quietMode = false;

	float ratio = 16.0f / 9.0f;
	int width = 256;
	int height = static_cast<int>(width / ratio);
	int samplePerPixel = 100;
	int recursionMaxDepth = 100;
	std::string sceneName;
	std::string saveFilePath;
	bool use_gpu = false;
	int nb_cpu_cores = 1;

	static render_parameters getArgs(int argc, char* argv[])
	{
		render_parameters params;

		int count;
		for (count = 0; count < argc; count++)
		{
			std::string arg = argv[count];

			if (arg[0] == '-')
			{
				std::string param = arg.substr(1);
				std::string value = argv[count + 1];

				if (param == "quiet")
				{
					params.quietMode = true;
				}
				else if (param == "width" && !value.empty())
				{
					params.width = stoul(value, 0, 10);
				}
				else if (param == "height" && !value.empty())
				{
					params.height = stoul(value, 0, 10);
				}
				else if (param == "ratio" && !value.empty())
				{
					double p1 = 0, p2 = 0;

					std::stringstream test(value);
					std::string segment;

					unsigned int loop = 0;
					while (getline(test, segment, ':'))
					{
						if (loop == 0)
						{
							p1 = stoul(segment, 0, 10);
						}
						else if (loop == 1)
						{
							p2 = stoul(segment, 0, 10);
						}

						loop++;
					}

					if (p1 > 0 && p2 > 0)
					{
						params.ratio = static_cast<float>(p1 / p2);

						if (params.ratio > 1)
							params.height = static_cast<unsigned int>(params.width / params.ratio);
						else
							params.width = static_cast<unsigned int>(params.height * params.ratio);
					}
				}
				else if (param == "spp" && !value.empty())
				{
					params.samplePerPixel = stoul(value, 0, 10);
				}
				else if (param == "maxdepth" && !value.empty())
				{
					params.recursionMaxDepth = stoul(value, 0, 10);
				}
				else if (param == "scene" && !value.empty())
				{
					params.sceneName = value;
				}
				else if (param == "mode" && !value.empty())
				{
					params.nb_cpu_cores = stoul(value, 0, 10);
				}
				else if (param == "save" && !value.empty())
				{
					params.saveFilePath = value;
				}
			}
		}

		return params;
	}
};
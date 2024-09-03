#pragma once

#include "../misc/color.cuh"

//#define WIN32_LEAN_AND_MEAN // Exclude rarely-used services from Windows headers
//#define NOMINMAX // Prevent the definition of min and max macros
//#include <windows.h>

class output
{
public:
	__host__ virtual int init_output(const size_t dataSize)
	{
		return 0;
	}
	
	__host__ virtual int write_to_output(int x, int y, color pixel_color) const
	{
		return 0;
	}

	__host__ virtual int clean_output()
	{
		return 0;
	}
};
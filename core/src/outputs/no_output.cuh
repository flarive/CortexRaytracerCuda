#pragma once

#include "output.cuh"

//#define WIN32_LEAN_AND_MEAN // Exclude rarely-used services from Windows headers
//#define NOMINMAX // Prevent the definition of min and max macros
//#include <windows.h>


class no_output : public output
{
public:
    __host__ int init_output(const size_t dataSize) override;
    __host__ int write_to_output(int x, int y, color pixel_color) const override;
    __host__ int clean_output() override;
};

__host__ int no_output::init_output(const size_t dataSize)
{
    return 0;
}

__host__ int no_output::write_to_output(int x, int y, color pixel_color) const
{
    return 0;
}

__host__ int no_output::clean_output()
{
    return 0;
}
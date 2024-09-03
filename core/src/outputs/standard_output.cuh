#pragma once

#include "output.cuh"

#include "../utilities/interval.cuh"

//#define WIN32_LEAN_AND_MEAN // Exclude rarely-used services from Windows headers
//#define NOMINMAX // Prevent the definition of min and max macros
//#include <windows.h>

class standard_output : public output
{
public:
    __host__ int init_output(const size_t dataSize) override;
    __host__ int write_to_output(int x, int y, color pixel_color) const override;
    __host__ int clean_output() override;
};

__host__ int standard_output::init_output(const size_t dataSize)
{
    return 0;
}

__host__ int standard_output::write_to_output(int x, int y, color pixel_color) const
{
    // Write the translated [0,255] value of each color component.
    // Static Variable gets constructed only once no matter how many times the function is called.
    static const interval intensity(0.000, 0.999);

    std::cout << "p " << x << " " << y << " "
        << static_cast<int>(256 * intensity.clamp(pixel_color.r())) << " "
        << static_cast<int>(256 * intensity.clamp(pixel_color.g())) << " "
        << static_cast<int>(256 * intensity.clamp(pixel_color.b())) << "\n";

    return 0;
}

__host__ int standard_output::clean_output()
{
    return 0;
}
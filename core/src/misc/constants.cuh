#pragma once

//#include <limits>
//#include <numbers>
//#include <cmath>
#define _USE_MATH_DEFINES
#include <math.h>



//#include <cuda_runtime.h>

// Constants
//const double infinity = HUGE_VAL;// std::numeric_limits<double>::infinity();
//const float M_PI = 3.1415926535897932385f; //3.141592653589793238462643383279502884 ???
//const double M_PI = std::numbers::pi;

//const float M_2_PI = M_PI * 2.0f;
//
//
//const double M_1_PI = 0.318309886183790671538;  // 1/pi
//
//const double M_PI_2 = 1.57079632679489661923;   // pi/2

const float SHADOW_ACNE_FIX = 0.00001f;

// to test !!!!!!!!!!!!!!
//const float SHADOW_BIAS = 1e-4;

__host__ __device__ inline float get_infinity()
{
	return HUGE_VAL;
}

__host__ __device__ inline float get_pi()
{
	return 3.14159274101257324219f;
}

__host__ __device__ inline float get_2_pi()
{
	return 6.28318530718f;
}

__host__ __device__ inline float get_1_div_pi()
{
	return 0.318309886183790671538f;
}

__host__ __device__ inline float get_half_pi()
{
	return 1.57079632679489661923f;
}

__host__ __device__ inline float get_shadow_acne_fix()
{
	return 0.00001f;
}
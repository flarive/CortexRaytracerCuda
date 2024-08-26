#pragma once

//#include <limits>
//#include <numbers>
//#include <cmath>
//#define _USE_MATH_DEFINES
//#include <math.h>



//#include <cuda_runtime.h>

// Constants
//__device__ const double INFINITY = HUGE_VAL;// std::numeric_limits<double>::infinity();
//const float M_PI = 3.1415926535897932385f; //3.141592653589793238462643383279502884 ???
//const double M_PI = std::numbers::pi;

//const float M_2_PI = M_PI * 2.0f;
//
//
//const double M_1_PI = 0.318309886183790671538;  // 1/pi
//
//const double M_PI_2 = 1.57079632679489661923;   // pi/2

__device__ const float M_DOUBLE_PI = 6.28318530718f;

__device__ const float SHADOW_ACNE_FIX = 0.00001f;


#define M_E        2.71828182845904523536f   // e
#define M_LOG2E    1.44269504088896340736f   // log2(e)
#define M_LOG10E   0.434294481903251827651f  // log10(e)
#define M_LN2      0.693147180559945309417f  // ln(2)
#define M_LN10     2.30258509299404568402f   // ln(10)
#define M_PI       3.14159265358979323846f   // pi
#define M_PI_2     1.57079632679489661923f   // pi/2
#define M_PI_4     0.785398163397448309616f  // pi/4
#define M_1_PI     0.318309886183790671538f  // 1/pi
#define M_2_PI     0.636619772367581343076f  // 2/pi
#define M_2_SQRTPI 1.12837916709551257390f   // 2/sqrt(pi)
#define M_SQRT2    1.41421356237309504880f   // sqrt(2)
#define M_SQRT1_2  0.707106781186547524401f  // 1/sqrt(2)



// to test !!!!!!!!!!!!!!
//const float SHADOW_BIAS = 1e-4;

//__host__ __device__ inline float get_infinity()
//{
//	return HUGE_VAL;
//}

//__host__ __device__ inline float get_pi()
//{
//	return 3.14159274101257324219f;
//}
//
//__host__ __device__ inline float get_2_pi()
//{
//	return 6.28318530718f;
//}
//
//__host__ __device__ inline float get_1_div_pi()
//{
//	return 0.318309886183790671538f;
//}
//
//__host__ __device__ inline float get_half_pi()
//{
//	return 1.57079632679489661923f;
//}

//__host__ __device__ inline float get_shadow_acne_fix()
//{
//	return 0.00001f;
//}

//__host__ __device__ inline float get_m_2_pi()
//{
//	return 0.636619772367581343076f;
//}

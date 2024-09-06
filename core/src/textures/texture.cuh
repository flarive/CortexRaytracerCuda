#pragma once

#include "perlin.cuh"


__host__ __device__ enum class TextureTypeID {
    textureBaseType = 0,
    textureSolidColorType = 1,
    textureImageType = 2,
	textureCheckerType = 3,
	texturePerlinNoiseType = 4,
    textureBumpType = 5,
    textureNormalType = 6,
    textureDisplacementType = 7,
	textureEmissiveType = 8,
    textureAlphaType = 9,
    textureGradientType = 10
};



class texture
{
public:
    __host__ __device__ virtual color value(float u, float v, const point3& p) const = 0;

    __host__ __device__ virtual TextureTypeID getTypeID() const { return TextureTypeID::textureBaseType; }
};
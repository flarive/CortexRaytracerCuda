#pragma once

#include "../misc/constants.cuh"
#include "../misc/vector3.cuh"
#include "../misc/color.cuh"

template<typename T>
__host__ __device__ T lerp(const T& a, const T& b, const T& x)
{
    // FMA friendly
    return x * b + (a - a * x);
}

/**
 * \brief Map a 3D point with a transformation.
 *        Convert the 3D vector to 4D homogeneous coordinates and back to 3D.
 * \param transformation A 4x4 transformation matrix
 * \param point A 3D point
 * \return The transformed 3D point
 */
__host__ __device__ static vector3 mapPoint(const matrix4& transformation, const vector3& point)
{
    const vector4 homogeneousPoint(point, 1.0);
    const auto homogeneousResult = transformation * homogeneousPoint;

    assert(homogeneousResult.w != 0.0);

    return vector3(homogeneousResult) / homogeneousResult.w;
}


/**
 * \brief Map a 3D vector with a transformation.
 *        Convert the 3D vector to 4D homogeneous coordinates and back to 3D.
 * \param transformation A 4x4 transformation matrix
 * \param vector A 3D vector
 * \return The transformed 3D vector
 */
__host__ __device__ static vector3 mapVector(const matrix4& transformation, const vector3& vector)
{
    const vector4 homogeneousVector(vector, 0.0);
    const auto homogeneousResult = transformation * homogeneousVector;
    // Conversion from vector4 to vector3
    return homogeneousResult;
}


__host__ __device__ static bool near_zero(vector3 v)
{
    // Return true if the vector is close to zero in all dimensions.
    auto s = 1e-8;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

__host__ __device__ inline float degrees_to_radians(float degrees)
{
    return degrees * M_PI / 180.0f;
}

/// <summary>
/// Print a vector in stdout
/// </summary>
/// <param name="out"></param>
/// <param name="v"></param>
/// <returns></returns>
__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const vector3& v)
{
    return out << v.x << ' ' << v.y << ' ' << v.z;
}

// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
__host__ __device__ static void get_barycenter(point3 p, point3 a, point3 b, point3 c, float& u, float& v, float& w)
{
    vector3 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = glm::dot(v0, v0);
    float d01 = glm::dot(v0, v1);
    float d11 = glm::dot(v1, v1);
    float d20 = glm::dot(v2, v0);
    float d21 = glm::dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    v = (d11 * d20 - d01 * d21) / denom;
    w = (d00 * d21 - d01 * d20) / denom;
    u = 1.0f - v - w;
}

/// <summary>
/// https://medium.com/@dbildibay/ray-tracing-adventure-part-iv-678768947371
/// </summary>
/// <param name="tan"></param>
/// <param name="bitan"></param>
/// <param name="normal"></param>
/// <param name="sampleNormal"></param>
/// <returns></returns>
__host__ __device__ static vector3 getTransformedNormal(const vector3& tan, const vector3& bitan, const vector3& normal, color& sample, float strength, bool useMatrix)
{
    if (useMatrix)
    {
        // Build a TNB matrix (Tangent/Normal/Bitangent matrix)
        glm::mat3x3 matTNB = glm::mat3x3(tan, bitan, normal);
        vector3 tmp = vector3(sample.r(), sample.g(), sample.b());

        // Apply TNB matrix transformation to the texture space normal
        vector3 transformed_normal = matTNB * tmp;

        // Scale the transformed normal by the normal_strength factor
        transformed_normal *= strength;

        // Normalize the scaled transformed normal to ensure it's a unit vector
        return transformed_normal;
    }
    else
    {
        // simplest method (often sufficient and easier to implement)
        return tan * (sample.r() * strength) + bitan * (sample.g() * strength) + normal * (sample.b() * strength);
    }
}

// Function to calculate the maximum of the dot product of two vectors and zero
__host__ __device__ static float maxDot3(const vector3& v1, const vector3& v2)
{
    float dotProduct = 0.0;

    // Compute the dot product of the two vectors
    for (auto i = 0; i < v1.length(); ++i)
    {
        dotProduct += v1[i] * v2[i];
    }

    // Return the maximum of the dot product and zero
    return glm::max(dotProduct, 0.0f);
}

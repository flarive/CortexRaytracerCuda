#pragma once

#include "material.cuh"
#include "../misc/ray.cuh"
#include "../primitives/hittable_list.cuh"
#include "../misc/gpu_randomizer.cuh"

/// <summary>
/// Phong material
/// https://stackoverflow.com/questions/24132774/trouble-with-phong-shading
/// </summary>
class phong : public material
{
public:

    __host__ __device__ phong(texture* diffuseTexture, texture* specularTexture, const color& ambientColor, float shininess);

    __host__ __device__ phong(texture* diffuseTexture, texture* specularTexture, texture* bumpTexture, texture* normalTexture, texture* displaceTexture, texture* alphaTexture, texture* emissiveTexture, const color& ambientColor, float shininess);

    __device__ bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const override;
    __host__ __device__ float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const override;

private:
    color m_ambientColor{};
    float m_shininess = 0.0f;
};



__host__ __device__ phong::phong(texture* diffuseTexture, texture* specularTexture, const color& ambientColor, float shininess) 
    : phong(diffuseTexture, specularTexture, nullptr, nullptr, nullptr, nullptr, nullptr, ambientColor, shininess)
{
}

__host__ __device__ phong::phong(texture* diffuseTexture, texture* specularTexture, texture* bumpTexture, texture* normalTexture, texture* displaceTexture, texture* alphaTexture, texture* emissiveTexture, const color& ambientColor, float shininess) 
    : material(diffuseTexture, specularTexture, normalTexture, bumpTexture, displaceTexture, alphaTexture, emissiveTexture)
{
    m_ambientColor = ambientColor;
    m_shininess = shininess;
}
__device__ bool phong::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, curandState* local_rand_state) const
{
    vector3 normalv = rec.normal;

    vector3 hit_point = rec.hit_point;

    color diffuse_color;
    color specular_color;
    color emissive_color(0, 0, 0); // Initialize emissive color to black

    // Get the texture color at the hit point (assuming diffuse texture)
    if (m_diffuse_texture)
    {
        diffuse_color = m_diffuse_texture->value(rec.u, rec.v, hit_point);
    }

    if (m_specular_texture)
    {
        specular_color = m_specular_texture->value(rec.u, rec.v, hit_point);
    }

    // just take the first light for the moment
    if (lights.object_count == 0)
    {
        // no light
        return false;
    }

    light* mylight = std::dynamic_pointer_cast<light>(lights.objects[0]);
    if (mylight == nullptr)
    {
        // no light
        return false;
    }

    // Find the direction to the light source
    vector3 dirToLight = glm::normalize(mylight->getPosition() - hit_point);

    color lightColor = mylight->getColor() * mylight->getIntensity();

    if (m_displacement_texture)
    {
        // not handled here ! see mesh_loader.cpp
    }

    if (m_bump_texture)
    {
        // Check if a bump map texture is available
        std::shared_ptr<bump_texture> bumpTex = std::dynamic_pointer_cast<bump_texture>(m_bump_texture);
        if (bumpTex)
        {
            normalv = bumpTex->perturb_normal(normalv, rec.u, rec.v, hit_point);
        }
    }
    else if (m_normal_texture)
    {
        // Check if a normal map texture is available
        std::shared_ptr<normal_texture> normalTex = std::dynamic_pointer_cast<normal_texture>(m_normal_texture);
        if (normalTex)
        {
            // Sample the normal map texture to get the perturbed normal
            color normal_map = m_normal_texture->value(rec.u, rec.v, hit_point);

            // Transform the perturbed normal from texture space to world space
            // Apply the normal strength factor to the perturbed normal
            normalv = getTransformedNormal(rec.tangent, rec.bitangent, normalv, normal_map, normalTex->getStrenth(), false);
        }
    }

    if (m_alpha_texture)
    {
        // Check if a alpha map texture is available
        std::shared_ptr<alpha_texture> alphaTex = std::dynamic_pointer_cast<alpha_texture>(m_alpha_texture);
        if (alphaTex)
        {
            // good idea ?
            srec.alpha_value = alphaTex->value(rec.u, rec.v, hit_point).r();
        }
    }

    if (m_emissive_texture)
    {
        // Check if a emissive map texture is available
        std::shared_ptr<emissive_texture> emissiveTex = std::dynamic_pointer_cast<emissive_texture>(m_emissive_texture);
        if (emissiveTex)
        {
            emissive_color = emissiveTex->value(rec.u, rec.v, hit_point);
        }
    }

    vector3 v = glm::normalize(-1.0 * (hit_point - r_in.origin()));
    double nl = maxDot3(normalv, dirToLight);
    vector3 r = glm::normalize((2.0 * nl * normalv) - dirToLight);


    // Combine the surface color with the light's color/intensity
    //color final_color = (diffuse_color * nl + specular_color * pow(maxDot3(v, r), m_shininess)) * lightColor;
    color final_color = (diffuse_color * nl + specular_color * pow(maxDot3(v, r), m_shininess)) * lightColor + emissive_color;


    // No refraction, only reflection
    srec.attenuation = final_color;
    srec.pdf_ptr = std::make_shared<sphere_pdf>();
    srec.skip_pdf = false;

    return true;
}

__host__ __device__ float phong::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
    auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
    return cos_theta < 0 ? 0 : cos_theta / M_PI;
}

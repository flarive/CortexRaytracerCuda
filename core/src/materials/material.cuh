#pragma once

#include "../misc/scatter_record.cuh"

#include "../primitives/hittable.cuh"
#include "../primitives/hittable_list.cuh"


__host__ __device__ enum class MaterialTypeID {
    materialBaseType = 0,
    materialLambertianType = 1,
    materialMetalType = 2,
    materialDielectricType = 3,
    materialIsotropicType = 4,
    materialAnisotropicType = 5,
    materialPhongType = 6,
    materialOrenNayarType = 7,
    materialDiffuseLightType = 8,
    materialDiffuseSpotLightType = 9
};


class material
{
public:
    //__host__ __device__ virtual color emitted(float u, float v, const vector3& p) const { return color(0,0,0); }
    //__host__ __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vector3& attenuation, ray& scattered, curandState *rng) const = 0;



    __host__ __device__ material()
        : material(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse)
        : material(_diffuse, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular)
        : material(_diffuse, _specular, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal)
        : material(_diffuse, _specular, _normal, nullptr, nullptr, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal, texture* _bump)
        : material(_diffuse, _specular, _normal, _bump, nullptr, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal, texture* _bump, texture* _displace)
        : material(_diffuse, _specular, _normal, _bump, _displace, nullptr, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal, texture* _bump, texture* _displace, texture* _alpha)
        : material(_diffuse, _specular, _normal, _bump, _displace, _alpha, nullptr, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal, texture* _bump, texture* _displace, texture* _alpha, texture* _emissive)
        : material(_diffuse, _specular, _normal, _bump, _displace, _alpha, _emissive, 0, 0)
    {
    }

    __host__ __device__ material(texture* _diffuse, texture* _specular, texture* _normal, texture* _bump, texture* _displace, texture* _alpha, texture* _emissive, float transparency, float refractive_index)
        : m_diffuse_texture(_diffuse), m_specular_texture(_specular), m_normal_texture(_normal), m_bump_texture(_bump), m_displacement_texture(_displace), m_alpha_texture(_alpha), m_emissive_texture(_emissive), m_transparency(transparency), m_refractiveIndex(refractive_index)
    {
    }

    __host__ __device__ virtual ~material();

    __host__ __device__ virtual MaterialTypeID getTypeID() const { return MaterialTypeID::materialBaseType; }

    __device__ virtual bool scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const;
    __host__ __device__ virtual float scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const;
    __device__ virtual color emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, thrust::default_random_engine& rng) const;

    __host__ __device__ bool has_alpha_texture(bool& double_sided) const;
    __host__ __device__ bool has_displace_texture() const;

    __host__ __device__ texture* get_diffuse_texture() const;
    __host__ __device__ color get_diffuse_pixel_color(const hit_record& rec) const;

    __host__ __device__ texture* get_displacement_texture() const;

protected:
    texture* m_diffuse_texture = nullptr;
    texture* m_specular_texture = nullptr;
    texture* m_normal_texture = nullptr;
    texture* m_bump_texture = nullptr;
    texture* m_displacement_texture = nullptr;
    texture* m_alpha_texture = nullptr;
    texture* m_emissive_texture = nullptr;

    //bool m_isTransparent = false;
    float m_refractiveIndex = 0.0f;
    float m_transparency = 0.0f;

    bool m_has_alpha = false;
    float m_alpha_value = 1.0f;

    //double m_reflectivity = 0;
    //double m_transparency = 0;
    //double m_refractiveIndex = 0;
    //double m_specularity = 0;
    //double m_specularExponent = 0;
    //double m_emissivity = 0;
    //double m_roughness = 0;
};



__device__ inline bool material::scatter(const ray& r_in, const hittable_list& lights, const hit_record& rec, scatter_record& srec, thrust::default_random_engine& rng) const
{
    return false;
}

__host__ __device__ inline float material::scattering_pdf(const ray& r_in, const hit_record& rec, const ray& scattered) const
{
    return 0;
}

__device__ inline color material::emitted(const ray& r_in, const hit_record& rec, float u, float v, const point3& p, thrust::default_random_engine& rng) const
{
    return color(0, 0, 0);
}

__host__ __device__ inline bool material::has_alpha_texture(bool& double_sided) const
{
    /*if (m_alpha_texture)
    {
        std::shared_ptr<alpha_texture> derived1 = std::dynamic_pointer_cast<alpha_texture>(m_alpha_texture);
        if (derived1)
        {
            double_sided = derived1->is_double_sided();
        }

        return true;
    }*/

    return false;
}

__host__ __device__ inline bool material::has_displace_texture() const
{
    /*if (m_displacement_texture)
    {
        std::shared_ptr<displacement_texture> derived1 = std::dynamic_pointer_cast<displacement_texture>(m_displacement_texture);
        if (derived1)
        {
            return true;
        }
    }*/

    return false;
}

__host__ __device__ inline texture* material::get_diffuse_texture() const
{
    return m_diffuse_texture;
}

__host__ __device__ inline texture* material::get_displacement_texture() const
{
    return m_displacement_texture;
}

__host__ __device__ inline color material::get_diffuse_pixel_color(const hit_record& rec) const
{
    auto diffuse_tex = this->get_diffuse_texture();
    if (diffuse_tex)
    {
        /*solid_color_texture derived1 = std::dynamic_pointer_cast<solid_color_texture>(diffuse_tex);
        if (derived1)
        {
            return derived1->get_color();
        }
        else
        {
            std::shared_ptr<image_texture> derived2 = std::dynamic_pointer_cast<image_texture>(diffuse_tex);
            if (derived2)
            {
                return derived2->value(rec.u, rec.v, rec.hit_point);
            }
        }*/
    }

    return color{};
}

// Destructor implementation
__host__ __device__ inline material::~material()
{
    if (m_diffuse_texture) {
        delete m_diffuse_texture;
        m_diffuse_texture = nullptr;
    }

    if (m_specular_texture) {
        delete m_specular_texture;
        m_specular_texture = nullptr;
    }

    if (m_normal_texture) {
        delete m_normal_texture;
        m_normal_texture = nullptr;
    }

    if (m_bump_texture) {
        delete m_bump_texture;
        m_bump_texture = nullptr;
    }

    if (m_displacement_texture) {
        delete m_displacement_texture;
        m_displacement_texture = nullptr;
    }

    if (m_alpha_texture) {
        delete m_alpha_texture;
        m_alpha_texture = nullptr;
    }

    if (m_emissive_texture) {
        delete m_emissive_texture;
        m_emissive_texture = nullptr;
    }
}
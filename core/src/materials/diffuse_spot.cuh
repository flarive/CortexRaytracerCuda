#pragma once

#include "material.cuh"
#include "../textures/texture.cuh"
#include "../textures/solid_color_texture.cuh"

class diffuse_spot : public material
{
public:
    diffuse_spot_light(std::shared_ptr<texture> emitTex, point3 pos, vector3 dir, double cutoff, double falloff, double intensity, bool invisible);

    color emitted(const ray& r_in, const hit_record& rec, double u, double v, const point3& p) const override;

private:
    std::shared_ptr<texture> m_emit = nullptr;
    point3 m_position{};
    double m_intensity = 1.0;
    bool m_invisible = true;
    bool m_directional = true;
    vector3 m_direction = vector3(0, 0, -1); // Default direction
    double m_cutoff = cos(degrees_to_radians(30.0)); // Cutoff angle in radians
    double m_falloff = 1.0; // Falloff exponent
};


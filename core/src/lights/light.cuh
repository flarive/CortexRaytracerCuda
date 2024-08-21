#pragma once

/// <summary>
/// Abstract class for lights
/// </summary>
class light : public hittable
{
public:
    __host__ __device__ light(point3 _position, float _intensity, color _color, bool _invisible, const char* _name = "Light");
    __host__ __device__ virtual ~light() = default;

    __host__ __device__ double getIntensity() const;
    __host__ __device__ color getColor() const;
    __host__ __device__ virtual point3 getPosition() const;


protected:
    point3 m_position{};
    material* m_mat;
    float m_intensity = 0.0;
    bool m_invisible = true;
    color m_color{};
};



__host__ __device__ light::light(point3 _position, float _intensity, color _color, bool _invisible, const char* _name)
    : m_position(_position), m_intensity(_intensity), m_color(_color), m_invisible(_invisible)
{
    setName(_name);
}

//aabb light::bounding_box() const
//{
//    return m_bbox;
//}

__host__ __device__ double light::getIntensity() const
{
    return m_intensity;
}

__host__ __device__ color light::getColor() const
{
    return m_color;
}

__host__ __device__ point3 light::getPosition() const
{
    return m_position;
}
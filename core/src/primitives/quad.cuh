#pragma once

class quad : public hittable
{
public:
    __host__ __device__ quad(const point3& _position, const vector3& _u, const vector3& _v, material* _mat, const char* _name = "Quad");
    __host__ __device__ quad(const point3& _position, const vector3& _u, const vector3& _v, material* _mat, const uvmapping& _mapping, const char* _name = "Quad");






    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;


    


    __device__ float pdf_value(const point3& origin, const vector3& v, curandState* local_rand_state) const override;


    /// <summary>
    /// Random special implementation for quad (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const point3& origin, curandState* local_rand_state) const override;


    __host__ __device__ void set_bounding_box();

    __host__ __device__ aabb bounding_box() const override;


    __host__ __device__ bool is_interior(float a, float b, hit_record& rec) const;



private:
    point3 m_position; // the lower-left corner
    vector3 m_u; // a vector representing the first side
    vector3 m_v; //  a vector representing the second side
    material* m_mat = nullptr;
    vector3 m_normal{};
    float m_d = 0.0;
    vector3 m_w{}; // The vector w is constant for a given quadrilateral, so we'll cache that value
    float m_area = 0.0;
};




__host__ __device__ quad::quad(const point3& _position, const vector3& _u, const vector3& _v, material* _mat, const char* _name)
    : m_position(_position), m_u(_u), m_v(_v), m_mat(_mat)
{
    auto n = glm::cross(m_u, m_v);
    m_normal = unit_vector(n);
    m_d = glm::dot(m_normal, m_position);
    m_w = n / glm::dot(n, n);

    m_area = vector_length(n);

    setName(_name);

    set_bounding_box();
}

__host__ __device__ quad::quad(const point3& _position, const vector3& _u, const vector3& _v, material* _mat, const uvmapping& _mapping, const char* _name)
    : m_position(_position), m_u(_u), m_v(_v), m_mat(_mat)
{
    auto n = glm::cross(m_u, m_v);
    m_normal = unit_vector(n);
    m_d = glm::dot(m_normal, m_position);
    m_w = n / glm::dot(n, n);

    m_area = vector_length(n);

    setName(_name);

    set_bounding_box();
}





__host__ __device__ void quad::set_bounding_box()
{
    m_bbox = aabb(m_position, m_position + m_u + m_v).pad();
}

__host__ __device__ aabb quad::bounding_box() const
{
    return m_bbox;
}

__device__ bool quad::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    auto denom = glm::dot(m_normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < 1e-8)
        return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (m_d - glm::dot(m_normal, r.origin())) / denom;
    if (!ray_t.contains(t))
        return false;

    // Determine the hit point lies within the planar shape using its plane coordinates.
    auto intersection = r.at(t);
    vector3 planar_hitpt_vector = intersection - m_position;
    auto alpha = glm::dot(m_w, glm::cross(planar_hitpt_vector, m_v));
    auto beta = glm::dot(m_w, glm::cross(m_u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec))
        return false;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t = t;
    rec.hit_point = intersection;
    rec.mat = m_mat;
    rec.set_face_normal(r, m_normal);

    // name of the primitive hit by the ray
    rec.name = m_name;
    rec.bbox = m_bbox;

    return true;
}

__device__ float quad::pdf_value(const point3& origin, const vector3& v, curandState* local_rand_state) const
{
    hit_record rec;

    if (!this->hit(ray(origin, v), interval(get_shadow_acne_fix(), get_infinity()), rec, 0, local_rand_state))
        return 0;

    auto distance_squared = rec.t * rec.t * vector_length_squared(v);
    auto cosine = fabs(dot(v, rec.normal) / vector_length(v));

    return distance_squared / (cosine * m_area);
}

/// <summary>
/// Random special implementation for quad (override base)
/// </summary>
/// <param name="origin"></param>
/// <returns></returns>
__device__ vector3 quad::random(const point3& origin, curandState* local_rand_state) const
{
    auto p = m_position
        + (get_real(local_rand_state) * m_u)
        + (get_real(local_rand_state) * m_v);

    return p - origin;
}


__host__ __device__ bool quad::is_interior(float a, float b, hit_record& rec) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.

    if ((a < 0) || (1 < a) || (b < 0) || (1 < b))
        return false;

    rec.u = a;
    rec.v = b;
    return true;
}

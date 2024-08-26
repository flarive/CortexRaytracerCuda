#pragma once

/// <summary>
/// Directional light
/// </summary>
class directional_light : public light
{
public:
    __host__ __device__ directional_light(const point3& _position, const vector3& _u, const vector3& _v, float _intensity, color _color, const char* _name = "QuadLight", bool _invisible = true);


    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;
    __device__ float pdf_value(const point3& origin, const vector3& v, curandState* local_rand_state) const override;

    /// <summary>
    /// Random special implementation for quad light (override base)
    /// </summary>
    /// <param name="origin"></param>
    /// <returns></returns>
    __device__ vector3 random(const point3& origin, curandState* local_rand_state) const override;

    __host__ __device__ void set_bounding_box();
    __host__ __device__ aabb bounding_box() const override;


    __host__ __device__ bool is_interior(float a, float b, hit_record& rec) const;

    __host__ __device__ HittableTypeID getTypeID() const override { return HittableTypeID::lightDirectionalType; }

    


private:
    vector3 m_u{}; // vector representing the first side of the quadrilateral
    vector3 m_v{}; //  vector representing the second side of the quadrilateral
    vector3 m_normal{}; // vector representing quadrilateral normal
    float D = 0.0f;
    vector3 w{}; // vector w is constant for a given quadrilateral, so we'll cache that value
    float area = 0.0f;
};


__host__ __device__ directional_light::directional_light(const point3& _position, const vector3& _u, const vector3& _v, float _intensity, color _color, const char* _name, bool _invisible)
    : light(_position, _intensity, _color, _invisible, _name), m_u(_u), m_v(_v)
{
    m_mat = new diffuse_light(m_color, _intensity, true, m_invisible);

    auto n = glm::cross(m_u, m_v);
    m_normal = unit_vector(n);
    D = glm::dot(m_normal, m_position);
    w = n / glm::dot(n, n);

    area = vector_length(n);

    set_bounding_box();
}

__host__ __device__ void directional_light::set_bounding_box()
{
    //m_bbox = aabb(m_position, m_position + m_u + m_v).pad();

    // Calculate the quad's corners
    point3 corner1 = m_position - 0.5f * (m_u + m_v);
    point3 corner2 = m_position + 0.5f * (m_u + m_v);

    m_bbox = aabb(corner1, corner2).pad();
}

__host__ __device__ aabb directional_light::bounding_box() const
{
    return m_bbox;
}

__device__ bool directional_light::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    float denom = glm::dot(m_normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (fabs(denom) < 1e-8)
    {
        return false;
    }

    // Return false if the hit point parameter t is outside the ray interval.
    float t = (D - glm::dot(m_normal, r.origin())) / denom;
    if (!ray_t.contains(t))
    {
        return false;
    }

    // Determine the hit point lies within the planar shape using its plane coordinates.
    auto intersection = r.at(t);
    vector3 planar_hitpt_vector = intersection - m_position;
    auto alpha = glm::dot(w, glm::cross(planar_hitpt_vector, m_v));
    auto beta = glm::dot(w, glm::cross(m_u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec))
    {
        return false;
    }


    //Singleton* singleton = Singleton::getInstance();
    //if (singleton)
    //{
    //    auto renderParams = singleton->value();
    //    if (m_invisible && depth == renderParams.recursionMaxDepth)
    //    {
    //        return false;
    //    }
    //}

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t = t;
    rec.hit_point = intersection;
    rec.mat = m_mat;
    rec.name = m_name;
    rec.bbox = m_bbox;
    rec.set_face_normal(r, m_normal);

    return true;
}

__host__ __device__ bool directional_light::is_interior(float a, float b, hit_record& rec) const
{
    // Given the hit point in plane coordinates, return false if it is outside the
    // primitive, otherwise set the hit record UV coordinates and return true.

    //if ((a < 0) || (1 < a) || (b < 0) || (1 < b))
    //{
    //    return false;
    //}

    //rec.u = a;
    //rec.v = b;
    //return true;

    if ((a < -0.5) || (0.5 < a) || (b < -0.5) || (0.5 < b))
    {
        return false;
    }

    rec.u = a + 0.5f; // shift to [0, 1] range
    rec.v = b + 0.5f; // shift to [0, 1] range
    return true;
}

__device__ float directional_light::pdf_value(const point3& origin, const vector3& v, curandState* local_rand_state) const
{
    hit_record rec;

    if (!this->hit(ray(origin, v), interval(SHADOW_ACNE_FIX, INFINITY), rec, 0, local_rand_state))
        return 0;

    auto distance_squared = rec.t * rec.t * vector_length_squared(v);
    auto cosine = fabs(dot(v, rec.normal) / vector_length(v));

    return distance_squared / (cosine * area);
}

/// <summary>
/// Random special implementation for quad light (override base)
/// </summary>
/// <param name="origin"></param>
/// <returns></returns>
__device__ vector3 directional_light::random(const point3& origin, curandState* local_rand_state) const
{
    auto p = m_position
        + (get_real(local_rand_state) - 0.5f) * m_u
        + (get_real(local_rand_state) - 0.5f) * m_v;

    return p - origin;
}
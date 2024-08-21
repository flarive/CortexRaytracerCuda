#pragma once

class xy_rect : public hittable
{
public:
    __host__ __device__ xy_rect(const char* _name = nullptr);
    __host__ __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat, const char* _name = nullptr);
    __host__ __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat, const uvmapping& _mapping, const char* _name = nullptr);

    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;

    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;
    __host__ __device__ aabb bounding_box() const override;

private:

    material* mp = nullptr;
    float x0 = 0, x1 = 0, y0 = 0, y1 = 0, k = 0;
};


__host__ __device__ xy_rect::xy_rect(const char* _name)
    : xy_rect(0, 10, 0, 10, 10, nullptr, uvmapping(), _name)
{
}

__host__ __device__ xy_rect::xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat, const char* _name)
    : xy_rect(_x0, _x1, _y0, _y1, k, mat, uvmapping(), _name)
{
}

__host__ __device__ xy_rect::xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material* mat, const uvmapping& _mapping, const char* _name)
    : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat)
{
    if (_name != nullptr)
        setName(_name);
    else
        setName(new char[7] {"XYRect"});

    m_mapping = _mapping;

    m_bbox = aabb(point3(x0, y0, k - 0.0001f), point3(x1, y1, k + 0.0001f));
}

__device__ bool xy_rect::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    double t = (k - r.origin().z) / r.direction().z;
    if (t < ray_t.min || t > ray_t.max)
        return false;
    double x = r.origin().x + t * r.direction().x;
    double y = r.origin().y + t * r.direction().y;
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    //rec.u = (x - x0) / (x1 - x0);
    //rec.v = (y - y0) / (y1 - y0);

    // UV coordinates
    get_xy_rect_uv(x, y, rec.u, rec.v, x0, x1, y0, y1, m_mapping);

    rec.t = t;
    rec.mat = mp;
    rec.name = m_name;
    rec.bbox = m_bbox;
    rec.hit_point = r.at(t);
    rec.normal = vector3(0, 0, 1);
    return true;
}

__device__ float xy_rect::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 xy_rect::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3(1, 0, 0);
}


__host__ __device__ aabb xy_rect::bounding_box() const
{
    return m_bbox;
}

//
//__device__ bool xy_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const
//{
//    float t = (k - r.origin().z) / r.direction().z;
//    if (t < t0 || t > t1)
//        return false;
//    float x = r.origin().x + t*r.direction().x;
//    float y = r.origin().y + t*r.direction().y;
//    if (x < x0 || x > x1 || y < y0 || y > y1)
//        return false;
//    rec.u = (x - x0) / (x1 - x0);
//    rec.v = (y - y0) / (y1 - y0);
//    rec.t = t;
//    vector3 outward_normal = vector3(0, 0, 1);
//    rec.set_face_normal(r, outward_normal);
//    rec.mat_ptr = mp;
//    rec.p = r.point(t);
//
//    return true;
//}












class xz_rect : public hittable
{
public:
    __host__ __device__ xz_rect(const char* _name = nullptr);
    __host__ __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat, const char* _name = nullptr);
    __host__ __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat, const uvmapping& _mapping, const char* _name = nullptr);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;

    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    __host__ __device__ aabb bounding_box() const override;

    __host__ __device__ virtual HittableTypeID getTypeID() const { return HittableTypeID::hittableAaRectType; }

private:
    material* mp = nullptr;
    float x0 = 0, x1 = 0, z0 = 0, z1 = 0, k = 0;
};




__host__ __device__ xz_rect::xz_rect(const char* _name)
    : xz_rect(0, 10, 0, 10, 10, nullptr, uvmapping(), _name)
{
}

__host__ __device__ xz_rect::xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat, const char* _name)
    : xz_rect(_x0, _x1, _z0, _z1, _k, mat, uvmapping(), _name)
{
}

__host__ __device__ xz_rect::xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material* mat, const uvmapping& _mapping, const char* _name)
    : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat)
{
    if (_name != nullptr)
        setName(_name);
    else
        setName(new char[7] {"XZRect"});

    m_mapping = _mapping;

    m_bbox = aabb(vector3(x0, k - 0.0001f, z0), vector3(x1, k + 0.0001f, z1));
}

__device__ bool xz_rect::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    double t = (k - r.origin().y) / r.direction().y;
    if (t < ray_t.min || t > ray_t.max)
        return false;
    double x = r.origin().x + t * r.direction().x;
    double z = r.origin().z + t * r.direction().z;
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    //rec.u = (x - x0) / (x1 - x0);
    //rec.v = (z - z0) / (z1 - z0);

    // UV coordinates
    get_xz_rect_uv(x, z, rec.u, rec.v, x0, x1, z0, z1, m_mapping);

    rec.t = t;
    rec.mat = mp;
    rec.name = m_name;
    rec.bbox = m_bbox;
    rec.hit_point = r.at(t);
    rec.normal = vector3(0, 1, 0);
    return true;
}

__host__ __device__ aabb xz_rect::bounding_box() const
{
    return m_bbox;
}

__device__ float xz_rect::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0f;
}

__device__ vector3 xz_rect::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3(1, 0, 0);
}

//__device__ bool xz_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const
//{
//    float t = (k - r.origin().y) / r.direction().y;
//    if (t < t0 || t > t1)
//        return false;
//    float x = r.origin().x + t*r.direction().x;
//    float z = r.origin().z + t*r.direction().z;
//    if (x < x0 || x > x1 || z < z0 || z > z1)
//        return false;
//    rec.u = (x - x0) / (x1 - x0);
//    rec.v = (z - z0) / (z1 - z0);
//    rec.t = t;
//    vector3 outward_normal = vector3(0, 1, 0);
//    rec.set_face_normal(r, outward_normal);
//    rec.mat_ptr = mp;
//    rec.p = r.point(t);
//
//    return true;
//}

















class yz_rect : public hittable
{
public:
    __host__ __device__ yz_rect(const char* _name = nullptr);
    __host__ __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat, const char* _name = nullptr);
    __host__ __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat, const uvmapping& _mapping, const char* _name = nullptr);

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const override;

    __device__ float pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const override;

    __device__ vector3 random(const vector3& o, curandState* local_rand_state) const override;

    __host__ __device__ aabb bounding_box() const override;

private:
    material* mp = nullptr;
    float y0 = 0, y1 = 0, z0 = 0, z1 = 0, k = 0;
};



__host__ __device__ yz_rect::yz_rect(const char* _name)
    : yz_rect(0, 10, 0, 10, 10, nullptr, uvmapping(), _name)
{
}

__host__ __device__ yz_rect::yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat, const char* _name)
    : yz_rect(_y0, _y1, _z0, _z1, _k, mat, uvmapping(), _name)
{
}

__host__ __device__ yz_rect::yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material* mat, const uvmapping& _mapping, const char* _name)
    : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat)
{
    if (_name != nullptr)
        setName(_name);
    else
        setName(new char[7] {"YZRect"});

    m_mapping = _mapping;

    m_bbox = aabb(vector3(k - 0.0001f, y0, z0), vector3(k + 0.0001f, y1, z1));
}


__device__ bool yz_rect::hit(const ray& r, interval ray_t, hit_record& rec, int depth, curandState* local_rand_state) const
{
    double t = (k - r.origin().x) / r.direction().x;
    if (t < ray_t.min || t > ray_t.max)
        return false;
    double y = r.origin().y + t * r.direction().y;
    double z = r.origin().z + t * r.direction().z;
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    /*rec.u = (y - y0) / (y1 - y0);
    rec.v = (z - z0) / (z1 - z0);*/

    // UV coordinates
    get_yz_rect_uv(y, z, rec.u, rec.v, y0, y1, z0, z1, m_mapping);

    rec.t = t;
    rec.mat = mp;
    rec.name = m_name;
    rec.bbox = m_bbox;
    rec.hit_point = r.at(t);
    rec.normal = vector3(1, 0, 0);
    return true;
}

__device__ float yz_rect::pdf_value(const point3& o, const vector3& v, curandState* local_rand_state) const
{
    return 0.0;
}

__device__ vector3 yz_rect::random(const vector3& o, curandState* local_rand_state) const
{
    return vector3(1, 0, 0);
}


__host__ __device__ aabb yz_rect::bounding_box() const
{
    return m_bbox;
}


//__device__ bool yz_rect::hit(const ray& r, float t0, float t1, hit_record& rec) const {
//    float t = (k - r.origin().x) / r.direction().x;
//    if (t < t0 || t > t1)
//        return false;
//    float y = r.origin().y + t*r.direction().y;
//    float z = r.origin().z + t*r.direction().z;
//    if (y < y0 || y > y1 || z < z0 || z > z1)
//        return false;
//    rec.u = (y - y0) / (y1 - y0);
//    rec.v = (z - z0) / (z1 - z0);
//    rec.t = t;
//    vector3 outward_normal = vector3(1, 0, 0);
//    rec.set_face_normal(r, outward_normal);
//    rec.mat_ptr = mp;
//    rec.p = r.point(t);
//
//    return true;
//}
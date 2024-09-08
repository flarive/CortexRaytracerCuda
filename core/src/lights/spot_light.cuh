#pragma once

/// <summary>
/// Spot light
/// </summary>
class spot_light : public light
{
public:
	__host__ __device__ spot_light(point3 position, vector3 direction, float cutoff, float falloff, float intensity, float radius, color rgb, const char* _name = "SpotLight", bool invisible = true);


	__host__ __device__ aabb bounding_box() const override;

	/// <summary>
	/// Logic of sphere ray hit detection
	/// </summary>
	/// <param name="r"></param>
	/// <param name="ray_t"></param>
	/// <param name="rec"></param>
	/// <returns></returns>
	__device__ bool hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const override;


	__device__ float pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const override;


	/// <summary>
	/// Random special implementation for sphere lights (override base)
	/// </summary>
	/// <param name="origin"></param>
	/// <returns></returns>
	__device__ vector3 random(const point3& o, curandState* local_rand_state) const override;


private:
	vector3 m_direction{};
	float m_cutoff = 0.0f;
	float m_falloff = 0.0f;
	float m_radius = 0.0f;
	float m_blur = 0.0f;
};



__host__ __device__ spot_light::spot_light(point3 position, vector3 direction, float cutoff, float falloff, float intensity, float radius, color rgb, const char* name, bool invisible)
	: light(position, intensity, rgb, invisible, name)
{
	m_direction = direction;
	m_radius = radius;
	m_cutoff = cos(degrees_to_radians(cutoff));
	m_falloff = falloff;

	m_mat = new diffuse_spot_light(new solid_color_texture(m_color), m_position, m_direction, m_cutoff, m_falloff, m_intensity, m_invisible);

	// calculate stationary sphere bounding box for ray optimizations
	vector3 rvec = vector3(m_radius);
	m_bbox = aabb(m_position - rvec, m_position + rvec);
}

__host__ __device__ aabb spot_light::bounding_box() const
{
	return m_bbox;
}

__device__ bool spot_light::hit(const ray& r, interval ray_t, hit_record& rec, int depth, int max_depth, curandState* local_rand_state) const
{
	point3 center = m_position;
	vector3 oc = r.origin() - center;
	auto a = vector_length_squared(r.direction());
	auto half_b = glm::dot(oc, r.direction());
	auto c = vector_length_squared(oc) - m_radius * m_radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	auto sqrtd = sqrt(discriminant);

	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrtd) / a;
	if (!ray_t.surrounds(root)) {
		root = (-half_b + sqrtd) / a;
		if (!ray_t.surrounds(root))
			return false;
	}

	// Hide light source
	if (m_invisible && depth == max_depth)
	{
		return false;
	}


	// number of hits encountered by the ray (only the nearest ?)
	rec.t = root;

	// point coordinate of the hit
	rec.hit_point = r.at(rec.t);

	// material of the hit object
	rec.mat = m_mat;

	// name of the primitive hit by the ray
	rec.name = m_name;
	rec.bbox = m_bbox;

	// set normal and front-face tracking
	vector3 outward_normal = (rec.hit_point - center) / m_radius;
	rec.set_face_normal(r, outward_normal);

	// UV coordinates
	const uvmapping mapping = uvmapping();
	get_sphere_uv(outward_normal, rec.u, rec.v, mapping);

	return true;
}

__device__ float spot_light::pdf_value(const point3& o, const vector3& v, int max_depth, curandState* local_rand_state) const
{
	// This method only works for stationary spheres.
	hit_record rec;
	if (!this->hit(ray(o, v), interval(SHADOW_ACNE_FIX, INFINITY), rec, 0, max_depth, local_rand_state))
		return 0;

	auto cos_theta_max = sqrt(1 - m_radius * m_radius / vector_length_squared(m_position - o));
	auto solid_angle = 2 * M_PI * (1 - cos_theta_max);

	return  1 / solid_angle;
}

__device__ vector3 spot_light::random(const point3& o, curandState* local_rand_state) const
{
	vector3 direction = m_position - o;
	auto distance_squared = vector_length_squared(direction);
	onb uvw;
	uvw.build_from_w(direction);
	return uvw.local(random_to_sphere(local_rand_state, m_radius, distance_squared));
}
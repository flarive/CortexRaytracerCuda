#pragma once

#include "math_utils.cuh"
#include "../misc/vector3.cuh"
#include "../misc/constants.cuh"


class uvmapping
{
public:
	__host__ __device__ uvmapping()
		: m_scale_u(1.0), m_scale_v(1.0), m_offset_u(0.0), m_offset_v(0.0), m_repeat_u(1.0), m_repeat_v(1.0)
	{
	}

	__host__ __device__ uvmapping(float scale_u, float scale_v)
		: m_scale_u(scale_u), m_scale_v(scale_v), m_offset_u(0.0), m_offset_v(0.0), m_repeat_u(1.0), m_repeat_v(1.0)
	{
	}

	__host__ __device__ uvmapping(float scale_u, float scale_v, float offset_u, float offset_v)
		: m_scale_u(scale_u), m_scale_v(scale_v), m_offset_u(offset_u), m_offset_v(offset_v), m_repeat_u(1.0), m_repeat_v(1.0)
	{
	}

	__host__ __device__ uvmapping(float scale_u, float scale_v, float offset_u, float offset_v, float repeat_u, float repeat_v)
		: m_scale_u(scale_u), m_scale_v(scale_v), m_offset_u(offset_u), m_offset_v(offset_v), m_repeat_u(repeat_u), m_repeat_v(repeat_v)
	{
	}

	__host__ __device__ float scale_u() const;
	__host__ __device__ float scale_v() const;

	__host__ __device__ float offset_u() const;
	__host__ __device__ float offset_v() const;

	__host__ __device__ float repeat_u() const;
	__host__ __device__ float repeat_v() const;


	__host__ __device__ void scale_u(float su);
	__host__ __device__ void scale_v(float sv);

	__host__ __device__ void offset_u(float ou);
	__host__ __device__ void offset_v(float ov);

	__host__ __device__ void repeat_u(float ru);
	__host__ __device__ void repeat_v(float rv);

private:
	float m_scale_u = 1.0;
	float m_scale_v = 1.0;
	float m_offset_u = 0.0;
	float m_offset_v = 0.0;
	float m_repeat_u = 0.0;
	float m_repeat_v = 0.0;
};

__host__ __device__ extern void get_spherical_uv(const point3& p, float& u, float& v);
__host__ __device__ extern void get_spherical_uv(const point3& p, float texture_width, float texture_height, float render_width, float render_height, float& u, float& v);
__host__ __device__ extern vector3 from_spherical_uv(float u, float v);



__host__ __device__ extern void get_sphere_uv(const point3& p, float& u, float& v, const uvmapping& mapping);
__host__ __device__ extern void get_torus_uv(const vector3& _p, vector3& _c, float& _u, float& _v, float _majorRadius, float _minorRadius, const uvmapping& mapping);
__host__ __device__ extern void get_cylinder_uv(const vector3& p, float& u, float& v, float radius, float height, const uvmapping& mapping);
__host__ __device__ extern void get_disk_uv(const vector3& p, float& u, float& v, float radius, const uvmapping& mapping);
__host__ __device__ extern void get_cone_uv(const vector3& p, float& u, float& v, float radius, float height, const uvmapping& mapping);

__host__ __device__ extern void get_xy_rect_uv(float x, float y, float& u, float& v, float x0, float x1, float y0, float y1, const uvmapping& mapping);
__host__ __device__ extern void get_xz_rect_uv(float x, float y, float& u, float& v, float x0, float x1, float y0, float y1, const uvmapping& mapping);
__host__ __device__ extern void get_yz_rect_uv(float y, float z, float& u, float& v, float y0, float y1, float z0, float z1, const uvmapping& mapping);

__host__ __device__ extern void get_triangle_uv(const vector3 hitpoint, float& u, float& v, const vector3 verts[3], const vector2 vert_uvs[3]);
__host__ __device__ extern vector2 calculateTextureCoordinate(vector2 uv0, vector2 uv1, vector2 uv2, const vector2& barycentricCoords);

__host__ __device__ extern void get_screen_uv(int x, int y, float texture_width, float texture_height, float render_width, float render_height, float& u, float& v);



__host__ __device__ float uvmapping::scale_u() const
{
	return m_scale_u;
}

__host__ __device__ float uvmapping::scale_v() const
{
	return m_scale_v;
}

__host__ __device__ float uvmapping::offset_u() const
{
	return m_offset_u;
}

__host__ __device__ float uvmapping::offset_v() const
{
	return m_offset_v;
}

__host__ __device__ float uvmapping::repeat_u() const
{
	return m_repeat_u;
}

__host__ __device__ float uvmapping::repeat_v() const
{
	return m_repeat_v;
}


__host__ __device__ void uvmapping::scale_u(float su)
{
	m_scale_u = su;
}

__host__ __device__ void uvmapping::scale_v(float sv)
{
	m_scale_v = sv;
}

__host__ __device__ void uvmapping::offset_u(float ou)
{
	m_offset_u = ou;
}

__host__ __device__ void uvmapping::offset_v(float ov)
{
	m_offset_v = ov;
}

__host__ __device__ void uvmapping::repeat_u(float ru)
{
	m_repeat_u = ru;
}

__host__ __device__ void uvmapping::repeat_v(float rv)
{
	m_repeat_v = rv;
}


__host__ __device__ void get_spherical_uv(const point3& p, float& u, float& v)
{
	// p: a given point on the sphere of radius one, centered at the origin.
	// u: returned value [0,1] of angle around the Y axis from X=-1.
	// v: returned value [0,1] of angle from Y=-1 to Y=+1.
	//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
	//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
	//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

	auto theta = acos(-p.y);
	auto phi = atan2(-p.z, p.x) + M_PI;

	u = phi / (2 * M_PI);
	v = theta / M_PI;
}

__host__ __device__ void get_spherical_uv(const point3& p, float texture_width, float texture_height, float render_width, float render_height, float& u, float& v)
{
	// p: a given point on the sphere of radius one, centered at the origin.
	// u: returned value [0,1] of angle around the Y axis from X=-1.
	// v: returned value [0,1] of angle from Y=-1 to Y=+1.
	//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
	//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
	//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

	auto theta = acos(-p.y);
	auto phi = atan2(-p.z, p.x) + M_PI;

	u = phi / (2 * M_PI);
	v = theta / M_PI;

	// Calculate aspect ratios
	float texture_aspect_ratio = texture_width / texture_height;
	float render_aspect_ratio = render_width / render_height;

	// Adjust u and v to maintain aspect ratio
	if (texture_aspect_ratio > render_aspect_ratio) {
		u *= render_aspect_ratio / texture_aspect_ratio;
	}
	else {
		v *= texture_aspect_ratio / render_aspect_ratio;
	}

	// Normalize u and v
	u = std::fmod(u, 1.0);
	if (u < 0.0) u += 1.0;

	v = std::fmod(v, 1.0);
	if (v < 0.0) v += 1.0;
}

__host__ __device__ vector3 from_spherical_uv(float u, float v)
{
	float phi = 2 * M_PI * u, theta = M_PI * v;
	// THIS IS SUPER WEIRD?? Used only (AND KEEP IT THAT WAY) for environment importance sampling
	phi -= M_PI;

	return vector3(cos(phi) * sin(theta), -cos(theta), -sin(phi) * sin(theta));
}



__host__ __device__ void get_sphere_uv(const point3& p, float& u, float& v, const uvmapping& mapping)
{
	// p: a given point on the sphere of radius one, centered at the origin.
	// u: returned value [0,1] of angle around the Y axis from X=-1.
	// v: returned value [0,1] of angle from Y=-1 to Y=+1.
	//     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
	//     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
	//     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

	// Calculate spherical coordinates theta and phi
	auto theta = acos(-p.y); // Angle from the positive y-axis
	auto phi = atan2(-p.z, p.x) + M_PI; // Angle in the xy-plane around the z-axis

	// Normalize theta and phi to [0, 1] for texture coordinates
	float s = phi / (2 * M_PI); // Normalize phi to [0, 1] (u-coordinate)
	float t = theta / M_PI;     // Normalize theta to [0, 1] (v-coordinate)

	// Apply texture repetition (tiling/repeating) to s and t
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_torus_uv(const vector3& p, vector3& c, float& u, float& v, float majorRadius, float minorRadius, const uvmapping& mapping)
{
	//double phi = atan2(p.y, p.x);
	//if (phi < 0) phi += 2 * get_pi(); // Ensure phi is in [0, 2*pi]

	//// Calculate the distance from the center of the torus in the xy-plane
	//double dxy = glm::length(vector2(p.x, p.y) - vector2(c.x, c.y)) - majorRadius;

	//// Calculate the angle around the torus
	//double theta = atan2(p.z, dxy);
	//if (theta < 0) theta += 2 * get_pi(); // Ensure theta is in [0, 2*pi]

	//// Map phi and theta to the range [0, 1] for u and v coordinates
	//double s = phi / (2 * get_pi());
	//double t = theta / (2 * get_pi());

	//// Apply texture repetition (tiling/repeating) to s and t
	//s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	//t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	//// Map normalized coordinates (s, t) to (u, v) texture space
	//u = mapping.scale_u() * s + mapping.offset_u();
	//v = mapping.scale_v() * t + mapping.offset_v();

	// Calculate the angle phi around the major radius in the xy-plane
	float phi = atan2(p.y, p.x);
	if (phi < 0) phi += 2 * M_PI; // Ensure phi is in [0, 2*pi]

	// Calculate the point on the major radius circle
	vector3 majorCirclePoint(c.x + majorRadius * cos(phi), c.y + majorRadius * sin(phi), c.z);

	// Calculate the vector from the major circle point to the current point
	vector3 minorVec = p - majorCirclePoint;

	// Calculate the angle theta around the minor radius
	float theta = atan2(minorVec.z, glm::length(vector2(minorVec.x, minorVec.y)));
	if (theta < 0) theta += 2 * M_PI; // Ensure theta is in [0, 2*pi]

	// Map phi and theta to the range [0, 1] for u and v coordinates
	float s = phi / (2 * M_PI);
	float t = theta / (2 * M_PI);

	// Apply texture repetition (tiling/repeating) to s and t
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Ensure s and t are within [0, 1]
	if (s < 0) s += 1.0;
	if (t < 0) t += 1.0;

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_cylinder_uv(const vector3& p, float& u, float& v, float radius, float height, const uvmapping& mapping)
{
	// Calculate the angle around the cylinder using atan2
	float theta = std::atan2(p.x, p.z);

	// Map the angle (theta) to the range [0, 1] for u coordinate (s)
	float s = 1.0 - (theta + M_PI) / (2.0 * M_PI); // Invert theta and map to [0, 1]

	// Calculate the vertical height (y-coordinate) relative to the cylinder's height
	float y = p.y;
	float t = (y + height / 2.0) / height; // Map y-coordinate to [0, 1] range

	// Apply texture repetition (tiling/repeating) to s and t
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_disk_uv(const vector3& p, float& u, float& v, float radius, const uvmapping& mapping)
{
	//// Calculate the angle around the disk using atan2
	//double theta = std::atan2(p.x, p.z);

	//// Map the angle (theta) to the range [0, 1] for u coordinate (s)
	//double s = 1.0 - (theta + get_pi()) / (2.0 * get_pi()); // Invert theta and map to [0, 1]

	//// Calculate the vertical height (phi) relative to the disk's radius
	//double phi = std::atan2(p.y, radius);

	//// Map the vertical height (phi) to the range [0, 1] for v coordinate (t)
	//double t = (phi + get_pi() / 2.0) / get_pi(); // Map phi to [0, 1] range

	//// Apply texture repetition (tiling/repeating) to s and t
	//s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	//t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	//// Map normalized coordinates (s, t) to (u, v) texture space
	//u = mapping.scale_u() * s + mapping.offset_u();
	//v = mapping.scale_v() * t + mapping.offset_v();


	// Ensure point p is within the disk's radius
	float x = p.x;
	float z = p.z;

	// Map x and z coordinates to the range [0, 1] based on the disk's radius
	float s = (x / (2.0 * radius)) + 0.5;
	float t = (z / (2.0 * radius)) + 0.5;

	// Apply texture repetition (tiling/repeating) to s and t
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_cone_uv(const vector3& p, float& u, float& v, float radius, float height, const uvmapping& mapping)
{
	// Calculate the angle around the cone using atan2
	float theta = atan2(p.x, p.z);

	// Map the angle (theta) to the range [0, 1] for u coordinate
	float s = (theta + M_PI) / (2 * M_PI);

	// Calculate the distance from the cone apex to the point
	float distance = sqrt(p.x * p.x + p.z * p.z);

	// Map the distance to the range [0, 1] for v coordinate
	float t = distance / radius; // Normalize distance by radius

	// Apply texture repetition (tiling/repeating) to s and t
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_xy_rect_uv(float x, float y, float& u, float& v, float x0, float x1, float y0, float y1, const uvmapping& mapping)
{
	// Calculate normalized coordinates (s, t) within the range [0, 1]
	float s = (x - x0) / (x1 - x0);
	float t = (y - y0) / (y1 - y0);

	// Apply tiling to the normalized coordinates
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_xz_rect_uv(float x, float z, float& u, float& v, float x0, float x1, float z0, float z1, const uvmapping& mapping)
{
	// Calculate normalized coordinates (s, t) within the range [0, 1]
	float s = (x - x0) / (x1 - x0);
	float t = (z - z0) / (z1 - z0);

	// Apply tiling to the normalized coordinates
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_yz_rect_uv(float y, float z, float& u, float& v, float y0, float y1, float z0, float z1, const uvmapping& mapping)
{
	// Calculate normalized coordinates (s, t) within the range [0, 1]
	float s = (y - y0) / (y1 - y0);
	float t = (z - z0) / (z1 - z0);

	// Apply tiling to the normalized coordinates
	s = fmod(s * mapping.repeat_u(), 1.0); // Apply tiling to s (u-axis)
	t = fmod(t * mapping.repeat_v(), 1.0); // Apply tiling to t (v-axis)

	// Map normalized coordinates (s, t) to (u, v) texture space
	u = mapping.scale_u() * s + mapping.offset_u();
	v = mapping.scale_v() * t + mapping.offset_v();
}

__host__ __device__ void get_triangle_uv(const vector3 hitpoint, float& u, float& v, const vector3 verts[3], const vector2 vert_uvs[3])
{
	// https://www.irisa.fr/prive/kadi/Cours_LR2V/Cours/RayTracing_Texturing.pdf
	// https://computergraphics.stackexchange.com/questions/7738/how-to-assign-calculate-triangle-texture-coordinates

	float u1 = 0.0, v1 = 0.0, w1 = 0.0;
	// get triangle hit point barycenter uvs coords
	get_barycenter(hitpoint, verts[0], verts[1], verts[2], u1, v1, w1);

	// translate uvs from 3d space to texture space
	vector2 uv = calculateTextureCoordinate(vert_uvs[0], vert_uvs[1], vert_uvs[2], vector2(u1, v1));

	u = uv.x;
	v = uv.y;
}


/// <summary>
/// Function to calculate texture coordinates using barycentric coordinates
/// </summary>
__host__ __device__ vector2 calculateTextureCoordinate(vector2 uv0, vector2 uv1, vector2 uv2, const vector2& barycentricCoords)
{
	float u = (barycentricCoords.x * uv0.x + barycentricCoords.y * uv1.x + (1.0f - barycentricCoords.x - barycentricCoords.y) * uv2.x);
	float v = (barycentricCoords.x * uv0.y + barycentricCoords.y * uv1.y + (1.0f - barycentricCoords.x - barycentricCoords.y) * uv2.y);

	// Apply texture repeat (wrap) behavior
	u = std::fmod(u, 1.0);
	v = std::fmod(v, 1.0);
	if (u < 0.0) u += 1.0;
	if (v < 0.0) v += 1.0;

	return vector2(u, v); // Return texture coordinates
}

/// <summary>
/// TODO ! Could be enhanced by using stb_resize probably !
/// </summary>
__host__ __device__ void get_screen_uv(int x, int y, float texture_width, float texture_height, float render_width, float render_height, float& u, float& v)
{
	// Calculate normalized coordinates (u, v) within the range [0, 1]
	// Normalize pixel coordinates to [0, 1] with proper floating-point division
	u = (x / texture_width) * (texture_width / render_width);
	v = 1.0 - ((y / texture_height) * (texture_height / render_height));
}
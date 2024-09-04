#pragma once

#include "../misc/vector3.cuh"

#include <iostream>

class color
{
public:
    float c[4];

    __host__ __device__ color() : c{ 0, 0, 0, 1 } {}

    __host__ __device__ color(float c0) : c{ c0, c0, c0, 1 } {}
    
    __host__ __device__ color(float c0, float c1, float c2) : c{ c0, c1, c2, 1 } {}

    __host__ __device__ color(float c0, float c1, float c2, float c3) : c{ c0, c1, c2, c3 } {}

    __host__ __device__ float r() const { return c[0]; }
    __host__ __device__ float g() const { return c[1]; }
    __host__ __device__ float b() const { return c[2]; }
    __host__ __device__ float a() const { return c[3]; }

    __host__ __device__ void r(float r) { c[0] = r; }
    __host__ __device__ void g(float g) { c[1] = g; }
    __host__ __device__ void b(float b) { c[2] = b; }
    __host__ __device__ void a(float a) { c[3] = a; }

    __host__ __device__ color operator-() const;
    __host__ __device__ float operator[](int i) const;
    __host__ __device__ float& operator[](int i);


    __host__ __device__ color& operator+=(const color& v);
    __host__ __device__ color& operator+=(float t);
    __host__ __device__ color& operator*=(float t);
    __host__ __device__ color& operator*=(const color& v);
    __host__ __device__ color& operator/=(float t);


    __host__ __device__ float length() const;
    __host__ __device__ float length_squared() const;

    __host__ __device__ static color white();
    __host__ __device__ static color black();
    __host__ __device__ static color red();
    __host__ __device__ static color green();
    __host__ __device__ static color blue();
    __host__ __device__ static color yellow();
    __host__ __device__ static color undefined();

    /// <summary>
    /// Write pixel color to the output stream with pixel sampling (antialiasing) and gamma correction
    /// </summary>
    /// <param name="out"></param>
    /// <param name="pixel_color"></param>
    /// <param name="samples_per_pixel"></param>
    //__host__ __device__ static color prepare_pixel_color(int x, int y, color pixel_color, int samples_per_pixel, bool gamma_correction);

    __host__ __device__ static color RGBtoHSV(color rgb);
    __host__ __device__ static color HSVtoRGB(color hsv);

    //__host__ __device__ static float linear_to_gamma(float linear_component);

    __host__ __device__ static color blend_colors(const color& front, const color& back, float alpha);

    __host__ __device__ static color blend_with_background(const color& background, const color& object_color, float alpha);

    __host__ __device__ bool isValidColor();

    //__host__ __device__ static bool custom_isnan(float x);
        
};


// Color Utility Functions

__host__ __device__ inline std::ostream& operator<<(std::ostream& out, const color& v)
{
    return out << v.c[0] << ' ' << v.c[1] << ' ' << v.c[2];
}

__host__ __device__ inline color operator+(const color& u, const color& v)
{
    return color(u.c[0] + v.c[0], u.c[1] + v.c[1], u.c[2] + v.c[2]);
}

__host__ __device__ inline color operator-(const color& u, const color& v)
{
    return color(u.c[0] - v.c[0], u.c[1] - v.c[1], u.c[2] - v.c[2]);
}

__host__ __device__ inline color operator*(const color& u, const color& v)
{
    return color(u.c[0] * v.c[0], u.c[1] * v.c[1], u.c[2] * v.c[2]);
}

__host__ __device__ inline color operator*(float t, const color& v)
{
    return color(t * v.c[0], t * v.c[1], t * v.c[2]);
}

__host__ __device__ inline color operator*(const color& v, float t)
{
    return t * v;
}

__host__ __device__ inline color operator/(color v, float t)
{
    return (1 / t) * v;
}

__host__ __device__ inline color color::operator-() const {
    return color(-c[0], -c[1], -c[2], c[3]);
}

__host__ __device__ inline float color::operator[](int i) const {
    return c[i];
}

__host__ __device__ inline float& color::operator[](int i) {
    return c[i];
}

__host__ __device__ inline color& color::operator+=(const color& v) {
    if (v.c[3] == 0) return *this;

    c[0] += v.c[0];
    c[1] += v.c[1];
    c[2] += v.c[2];
    c[3] = glm::min(c[3] + v.c[3], 1.0f);

    return *this;
}

__host__ __device__ inline color& color::operator+=(float t) {
    c[0] += t;
    c[1] += t;
    c[2] += t;
    return *this;
}

__host__ __device__ inline color& color::operator*=(float t) {
    c[0] *= t;
    c[1] *= t;
    c[2] *= t;
    return *this;
}

__host__ __device__ inline color& color::operator*=(const color& v) {
    c[0] *= v[0];
    c[1] *= v[1];
    c[2] *= v[2];
    return *this;
}

__host__ __device__ inline color& color::operator/=(float t) {
    return *this *= 1.0f / t;
}



__host__ __device__ inline float color::length() const {
    return sqrt(length_squared());
}

__host__ __device__ inline float color::length_squared() const {
    return c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
}

__host__ __device__ inline color color::white() {
    return color(1, 1, 1);
}

__host__ __device__ inline color color::black() {
    return color(0, 0, 0);
}

__host__ __device__ inline color color::red() {
    return color(1, 0, 0);
}

__host__ __device__ inline color color::green() {
    return color(0, 1, 0);
}

__host__ __device__ inline color color::blue() {
    return color(0, 0, 1);
}

__host__ __device__ inline color color::yellow() {
    return color(1, 1, 0);
}

__host__ __device__ inline color color::undefined() {
    return color(-1, -1, -1);
}

__host__ __device__ inline static float linear_to_gamma(float linear_component) {
    return glm::sqrt(linear_component);
}

__host__ __device__ inline color color::blend_colors(const color& front, const color& back, float alpha) {
    return alpha * front + (1.0f - alpha) * back;
}

__host__ __device__ inline color color::blend_with_background(const color& background, const color& object_color, float alpha) {
    return (1.0f - alpha) * background + alpha * object_color;
}



template<class T>
__host__ __device__ inline T ffmin(T a, T b) { return(a < b ? a : b); }

template<class T>
__host__ __device__ inline T ffmax(T a, T b) { return(a > b ? a : b); }


__host__ __device__ inline color color::RGBtoHSV(color rgb)
{
    float max_val = ffmax(ffmax(rgb.r(), rgb.g()), rgb.b());
    float min_val = ffmin(ffmin(rgb.r(), rgb.g()), rgb.b());
    float delta_val = max_val - min_val;
    color hsv(0, delta_val > 0 ? delta_val / max_val : 0, max_val);

    if (delta_val > 0) {
        if (max_val == rgb.r()) {
            hsv.r(60 * std::fmodf((rgb.g() - rgb.b()) / delta_val, 6));
        }
        else if (max_val == rgb.g()) {
            hsv.r(60 * ((rgb.b() - rgb.r()) / delta_val + 2));
        }
        else if (max_val == rgb.b()) {
            hsv.r(60 * ((rgb.r() - rgb.g()) / delta_val + 4));
        }
        if (hsv.r() < 0) hsv.r(hsv.r() + 360);
    }

    return hsv;
}

__host__ __device__ inline color color::HSVtoRGB(color hsv)
{
    float chroma = hsv.b() * hsv.g();
    float fHPrime = std::fmodf(hsv.r() / 60.0f, 6);
    float x_val = chroma * (1 - std::fabs(std::fmodf(fHPrime, 2) - 1));
    float m_val = hsv.b() - chroma;

    color rgb;

    if (0 <= fHPrime && fHPrime < 1) {
        rgb = color(chroma, x_val, 0);
    }
    else if (1 <= fHPrime && fHPrime < 2) {
        rgb = color(x_val, chroma, 0);
    }
    else if (2 <= fHPrime && fHPrime < 3) {
        rgb = color(0, chroma, x_val);
    }
    else if (3 <= fHPrime && fHPrime < 4) {
        rgb = color(0, x_val, chroma);
    }
    else if (4 <= fHPrime && fHPrime < 5) {
        rgb = color(x_val, 0, chroma);
    }
    else if (5 <= fHPrime && fHPrime < 6) {
        rgb = color(chroma, 0, x_val);
    }
    else {
        rgb = color(0, 0, 0);
    }

    rgb += m_val;
    return rgb;
}

__host__ __device__ inline bool color::isValidColor()
{
    return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[3] >= 0;
}

__host__ __device__ inline static bool custom_isnan(float x)
{
    return x != x;
}

__host__ __device__ inline static color prepare_pixel_color(int x, int y, color pixel_color, int samples_per_pixel, bool gamma_correction)
{
    float r = custom_isnan(pixel_color.r()) ? 0.0f : pixel_color.r();
    float g = custom_isnan(pixel_color.g()) ? 0.0f : pixel_color.g();
    float b = custom_isnan(pixel_color.b()) ? 0.0f : pixel_color.b();

    if (samples_per_pixel > 0) {
        float scale = 1.0f / samples_per_pixel;
        r *= scale;
        g *= scale;
        b *= scale;
    }

    if (gamma_correction) {
        r = linear_to_gamma(r);
        g = linear_to_gamma(g);
        b = linear_to_gamma(b);
    }

    return color(r, g, b);
}


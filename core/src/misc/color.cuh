#pragma once

#include "../misc/vector3.cuh"

#include <iostream>

class color
{
public:
    double c[4];

    __device__ color() : c{ 0, 0, 0, 1 } {}

    __device__ color(float c0) : c{ c0, c0, c0, 1 } {}
    
    __device__ color(float c0, float c1, float c2) : c{ c0, c1, c2, 1 } {}

    __device__ color(float c0, float c1, float c2, float c3) : c{ c0, c1, c2, c3 } {}

    __device__ float r() const { return c[0]; }
    __device__ float g() const { return c[1]; }
    __device__ float b() const { return c[2]; }
    __device__ float a() const { return c[3]; }

    __device__ void r(float r) { c[0] = r; }
    __device__ void g(float g) { c[1] = g; }
    __device__ void b(float b) { c[2] = b; }
    __device__ void a(float a) { c[3] = a; }

    __device__ color operator-() const;
    __device__ float operator[](int i) const;
    __device__ double& operator[](int i);

    __device__ color& operator+=(const color& v);
    __device__ color& operator+=(float t);
    __device__ color& operator*=(float t);
    __device__ color& operator*=(const color& v);
    __device__ color& operator/=(float t);


    __device__ float length() const;
    __device__ float length_squared() const;

    __device__ static color white();
    __device__ static color black();
    __device__ static color red();
    __device__ static color green();
    __device__ static color blue();
    __device__ static color yellow();
    __device__ static color undefined();

    /// <summary>
    /// Write pixel color to the output stream with pixel sampling (antialiasing) and gamma correction
    /// </summary>
    /// <param name="out"></param>
    /// <param name="pixel_color"></param>
    /// <param name="samples_per_pixel"></param>
    __device__ static color prepare_pixel_color(int x, int y, color pixel_color, int samples_per_pixel, bool gamma_correction);

    __device__ static color RGBtoHSV(color rgb);
    __device__ static color HSVtoRGB(color hsv);

    __device__ static double linear_to_gamma(float linear_component);

    __device__ static color blend_colors(const color& front, const color& back, float alpha);

    __device__ static color blend_with_background(const color& background, const color& object_color, float alpha);

    __device__ bool isValidColor();
};


// Color Utility Functions

inline std::ostream& operator<<(std::ostream& out, const color& v)
{
    return out << v.c[0] << ' ' << v.c[1] << ' ' << v.c[2];
}

inline color operator+(const color& u, const color& v)
{
    return color(u.c[0] + v.c[0], u.c[1] + v.c[1], u.c[2] + v.c[2]);
}

inline color operator-(const color& u, const color& v)
{
    return color(u.c[0] - v.c[0], u.c[1] - v.c[1], u.c[2] - v.c[2]);
}

inline color operator*(const color& u, const color& v)
{
    return color(u.c[0] * v.c[0], u.c[1] * v.c[1], u.c[2] * v.c[2]);
}

inline color operator*(double t, const color& v)
{
    return color(t * v.c[0], t * v.c[1], t * v.c[2]);
}

inline color operator*(const color& v, float t)
{
    return t * v;
}

inline color operator/(color v, float t)
{
    return (1 / t) * v;
}



color color::operator-() const {
    return color(-c[0], -c[1], -c[2], c[3]);
}

float color::operator[](int i) const {
    return c[i];
}

double& color::operator[](int i) {
    return c[i];
}

color& color::operator+=(const color& v) {
    if (v.c[3] == 0) return *this;

    c[0] += v.c[0];
    c[1] += v.c[1];
    c[2] += v.c[2];
    c[3] = std::min(c[3] + v.c[3], 1.0);

    return *this;
}

color& color::operator+=(float t) {
    c[0] += t;
    c[1] += t;
    c[2] += t;
    return *this;
}

color& color::operator*=(float t) {
    c[0] *= t;
    c[1] *= t;
    c[2] *= t;
    return *this;
}

color& color::operator*=(const color& v) {
    c[0] *= v[0];
    c[1] *= v[1];
    c[2] *= v[2];
    return *this;
}

color& color::operator/=(float t) {
    return *this *= 1.0 / t;
}



float color::length() const {
    return sqrt(length_squared());
}

float color::length_squared() const {
    return c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
}

color color::white() {
    return color(1, 1, 1);
}

color color::black() {
    return color(0, 0, 0);
}

color color::red() {
    return color(1, 0, 0);
}

color color::green() {
    return color(0, 1, 0);
}

color color::blue() {
    return color(0, 0, 1);
}

color color::yellow() {
    return color(1, 1, 0);
}

color color::undefined() {
    return color(-1, -1, -1);
}

double color::linear_to_gamma(float linear_component) {
    return std::sqrt(linear_component);
}

color color::blend_colors(const color& front, const color& back, float alpha) {
    return alpha * front + (1.0 - alpha) * back;
}

color color::blend_with_background(const color& background, const color& object_color, float alpha) {
    return (1.0f - alpha) * background + alpha * object_color;
}



template<class T>
inline T ffmin(T a, T b) { return(a < b ? a : b); }

template<class T>
inline T ffmax(T a, T b) { return(a > b ? a : b); }


color color::RGBtoHSV(color rgb)
{
    float max_val = ffmax(ffmax(rgb.r(), rgb.g()), rgb.b());
    float min_val = ffmin(ffmin(rgb.r(), rgb.g()), rgb.b());
    float delta_val = max_val - min_val;
    color hsv(0, delta_val > 0 ? delta_val / max_val : 0, max_val);

    if (delta_val > 0) {
        if (max_val == rgb.r()) {
            hsv.r(60 * std::fmod((rgb.g() - rgb.b()) / delta_val, 6));
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

color color::HSVtoRGB(color hsv)
{
    float chroma = hsv.b() * hsv.g();
    float fHPrime = std::fmod(hsv.r() / 60.0, 6);
    float x_val = chroma * (1 - std::fabs(std::fmod(fHPrime, 2) - 1));
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

bool color::isValidColor()
{
    return c[0] >= 0 && c[1] >= 0 && c[2] >= 0 && c[3] >= 0;
}

color color::prepare_pixel_color(int x, int y, color pixel_color, int samples_per_pixel, bool gamma_correction)
{
    float r = std::isnan(pixel_color.r()) ? 0.0 : pixel_color.r();
    float g = std::isnan(pixel_color.g()) ? 0.0 : pixel_color.g();
    float b = std::isnan(pixel_color.b()) ? 0.0 : pixel_color.b();

    if (samples_per_pixel > 0) {
        float scale = 1.0 / samples_per_pixel;
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

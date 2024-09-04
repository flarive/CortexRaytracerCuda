#pragma once

class bitmap_image
{
public:

    //__host__ __device__ bitmap_image() {}
    __host__ __device__ bitmap_image(unsigned char* data, const int width, const int height, const int channels) 
        : m_data(data), m_image_width(width), m_image_height(height), m_image_channels(channels), m_bytes_per_scanline(width * channels)
    {
    }

    __host__ __device__ ~bitmap_image() {}




       
    __host__ __device__ int width()  const
    {
        return (m_data == nullptr) ? 0 : m_image_width;
    }

    __host__ __device__ int height() const
    {
        return (m_data == nullptr) ? 0 : m_image_height;
    }

    __host__ __device__ int channels() const
    {
        return (m_data == nullptr) ? 0 : m_bytes_per_pixel;
    }

    __host__ __device__ unsigned char* get_data() const
    {
        return m_data;
    }


    __host__ __device__ float* get_data_float() const;

    __host__ __device__ const unsigned char* pixel_data(int x, int y) const;


private:
    const int m_bytes_per_pixel = 3;
    unsigned char* m_data = NULL;
    int m_image_width = 0;
    int m_image_height = 0;
    int m_image_channels = 0;
    int m_bytes_per_scanline = 0;

    __host__ __device__ static int clamp_int(int x, int low, int high);
};



__host__ __device__ inline float* bitmap_image::get_data_float() const
{
    size_t numElements = m_image_width * m_image_height * m_bytes_per_pixel;
    float* floatArray = new float[numElements];
    for (size_t i = 0; i < numElements; ++i)
    {
        floatArray[i] = static_cast<float>(m_data[i]) / 255.0f; // Normalize to [0, 1]
    }

    // TODO free memory !
    //stbi_image_free(image);
    //delete[] floatImage;

    return floatArray;
}

__host__ __device__ inline const unsigned char* bitmap_image::pixel_data(int x, int y) const
{
    // Return the address of the three bytes of the pixel at x,y (or magenta if no data).
    static unsigned char magenta[] = { 255, 0, 255 };
    if (m_data == nullptr) return magenta;

    x = clamp_int(x, 0, m_image_width);
    y = clamp_int(y, 0, m_image_height);

    return m_data + y * m_bytes_per_scanline + x * m_bytes_per_pixel;
}

__host__ __device__ inline int bitmap_image::clamp_int(int x, int low, int high)
{
    // Return the value clamped to the range [low, high).
    if (x < low) return low;
    if (x < high) return x;
    return high - 1;
}
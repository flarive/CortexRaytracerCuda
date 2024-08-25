#pragma once

#include <cuda_runtime.h>
#include <cstring>

__host__ __device__ inline size_t stringlength(const char* s)
{
    size_t i = 0;
    while (s && *s != '\0') {
        s++;
        i++;
    }
    return i;
}

__host__ __device__ inline void strcpy2(const char* s, char* t)
{
    while (*s != '\0')
    {
        *(t) = *(s);
        s++;
        t++;
    }
    *t = '\0';
}

__host__ __device__ inline const char* concat(const char* str1, const char* str2)
{
    // Calculate the total length needed
    size_t len1 = stringlength(str1);
    size_t len2 = stringlength(str2);
    size_t totalLen = len1 + len2 + 1; // +1 for null terminator

    // Allocate memory for the new string
    char* result = new char[totalLen];

    // Copy the first string
    strcpy2(str1, result);

    // Copy the second string
    strcpy2(str2, result + len1);

    // Return the concatenated string
    return result;
}
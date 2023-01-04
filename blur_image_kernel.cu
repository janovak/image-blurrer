#include <stdint.h>
#include "index_helpers.cuh"

#define BLUR_RADIUS 25

extern "C" __global__ void BoxBlur(uint8_t *in_array, uint8_t *out_array, unsigned int width, unsigned int height)
{
    unsigned int sum = 0;
    unsigned int denominator = 0;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, width) + blockIdx.z;
#pragma unroll
    for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy)
    {
        const int neighborY = y + dy;
#pragma unroll
        for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, BLUR_RADIUS) + blockIdx.z;
            sum += in_array[neighborIndex];
            ++denominator;
        }
    }
    out_array[index] = sum / denominator;
}

extern "C" __global__ void GaussianBlur(uint8_t *in_array, uint8_t *out_array, float *gaussianKernel, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    float sum = 0;
    const unsigned int index = GetPixel(x, y, width) + blockIdx.z;
#pragma unroll
    for (int dy = -BLUR_RADIUS; dy <= BLUR_RADIUS; ++dy)
    {
        const int neighborY = y + dy;
#pragma unroll
        for (int dx = -BLUR_RADIUS; dx <= BLUR_RADIUS; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, BLUR_RADIUS) + blockIdx.z;
            sum += in_array[neighborIndex] * gaussianKernel[(dy + BLUR_RADIUS) * (BLUR_RADIUS * 2 + 1) + dx + BLUR_RADIUS];
        }
    }
    out_array[index] = static_cast<uint8_t>(sum);
}
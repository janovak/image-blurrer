#include <stdint.h>

#define RADIUS 25

__device__ unsigned int GetPixelWithPadding(unsigned int x, unsigned int y, unsigned int width, unsigned int padding)
{
    return (x + padding + (y + padding) * (width + 2 * padding)) * gridDim.z + blockIdx.z;
}

__device__ unsigned int GetPixel(unsigned int x, unsigned int y, unsigned int width)
{
    return GetPixelWithPadding(x, y, width, 0);
}

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
    const unsigned int index = GetPixel(x, y, width);
#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy)
    {
        const int neighborY = y + dy;
#pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, RADIUS);
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
    const unsigned int index = GetPixel(x, y, width);
#pragma unroll
    for (int dy = -RADIUS; dy <= RADIUS; ++dy)
    {
        const int neighborY = y + dy;
#pragma unroll
        for (int dx = -RADIUS; dx <= RADIUS; ++dx)
        {
            const int neighborX = x + dx;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, RADIUS);
            sum += in_array[neighborIndex] * gaussianKernel[(dy + RADIUS) * (RADIUS * 2 + 1) + dx + RADIUS];
        }
    }
    out_array[index] = static_cast<uint8_t>(sum);
}
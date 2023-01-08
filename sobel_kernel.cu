#include <algorithm>
#include <cmath>
#include <stdio.h>
#include "index_helpers.cuh"

#define FILTER_RADIUS 1
#define FILTER_WIDTH (FILTER_RADIUS * 2 + 1)

__device__ static const int HorizontalFilter[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
__device__ static const int VerticalFilter[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

extern "C" __global__ void SobelFilter(uint8_t *in_array, int16_t *out_array, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, width) + blockIdx.z;
    int dx = 0;
    int dy = 0;
#pragma unroll
    for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i)
    {
        const int neighborY = y + i;
#pragma unroll
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j)
        {
            const int neighborX = x + j;
            const unsigned int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, FILTER_RADIUS) + blockIdx.z;
            dx += in_array[neighborIndex] * HorizontalFilter[i + FILTER_RADIUS][j + FILTER_RADIUS];
            dy += in_array[neighborIndex] * VerticalFilter[i + FILTER_RADIUS][j + FILTER_RADIUS];
        }
    }
    out_array[index] = sqrt(pow(dx, 2) + pow(dy, 2));
}

extern "C" __global__ void NormalizeColors(int16_t *in_array, uint8_t *out_array, unsigned int width, unsigned int height, float min, float max)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, width) + blockIdx.z;
    out_array[index] = (in_array[index] - min) / (max - min) * 255;
}
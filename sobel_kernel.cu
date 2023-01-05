#include <cmath>
#include "index_helpers.cuh"

__device__ const int HorizontalFilter[3][3] = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
__device__ const int VerticalFilter[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

extern "C" __global__ void SobelFilter(uint8_t *in_array, uint8_t *out_array, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int index = GetPixel(x, y, width);
    if (x >= width || y >= height)
    {
        return;
    }
    int horizontalSum = 0;
    int verticalSum = 0;
#pragma unroll
    for (int i = -1; i <= 1; ++i)
    {
        const unsigned int neighborY = y + i;
#pragma unroll
        for (int j = -1; j <= 1; ++j)
        {
            const signed neighborX = x + j;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, 1) + blockIdx.z;
            horizontalSum += in_array[neighborIndex] * HorizontalFilter[i][j];
            verticalSum += in_array[neighborIndex] * VerticalFilter[i][j];
        }
    }
    out_array[index] = sqrt(pow(horizontalSum, 2) + pow(verticalSum, 2));
}
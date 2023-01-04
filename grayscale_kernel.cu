#include "index_helpers.cuh"

extern "C" __global__ void GrayscaleFilter(uint8_t *in_array, uint8_t *out_array, float *gaussianKernel, unsigned int width, unsigned int height)
{
    unsigned int sum = 0;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int index = GetPixel(x, y, width);
    if (x >= width || y >= height)
    {
        return;
    }
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        sum += in_array[index + i];
    }
    out_array[index] = sum / 3;
    out_array[index + 1] = sum / 3;
    out_array[index + 2] = sum / 3;
}
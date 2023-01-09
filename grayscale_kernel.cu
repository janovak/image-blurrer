#include "index_helpers.cuh"

extern "C" __global__ void GrayscaleFilter(uint8_t *in_array, uint8_t *out_array, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    // Multiply by 3 because gridDim.z should be 1, but we still have RGB values for each pixel
    const unsigned int index = GetPixel(x, y, width) * 3;
    unsigned int sum = 0;
    sum += in_array[index];
    sum += in_array[index + 1];
    sum += in_array[index + 2];
    out_array[index] = sum / 3;
    out_array[index + 1] = sum / 3;
    out_array[index + 2] = sum / 3;
}
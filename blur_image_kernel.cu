#define RADIUS 25

__device__ unsigned int GetPixelWithPadding(unsigned int x, unsigned int y, unsigned int width, unsigned int padding)
{
    return (x + padding + (y + padding) * (width + 2 * padding)) * gridDim.z + blockIdx.z;
}

__device__ unsigned int GetPixel(unsigned int x, unsigned int y, unsigned int width)
{
    return GetPixelWithPadding(x, y, width, 0);
}

// radius can't be an unsigned int because "-radius" is used in the for...loops
extern "C" __global__ void BoxBlur(int *in_array, int *out_array, unsigned int width, unsigned int height)
{
    unsigned int denominator = 0;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int index = GetPixel(x, y, width);
    out_array[index] = 0;
#pragma unroll
    for (int dx = -RADIUS; dx <= RADIUS; ++dx)
    {
#pragma unroll
        for (int dy = -RADIUS; dy <= RADIUS; ++dy)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;
            const int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, RADIUS);
            out_array[index] += in_array[neighborIndex];
            ++denominator;
        }
    }
    out_array[index] /= denominator;
}
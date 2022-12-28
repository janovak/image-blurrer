__device__ unsigned int GetPixel(unsigned int x, unsigned int y, unsigned int width)
{
    return (x + y * width) * gridDim.z + blockIdx.z;
}

// radius can't be an unsigned int because "-radius" is used in the for...loops
extern "C" __global__ void BoxBlur(int *in_array, int *out_array, int radius, unsigned int width, unsigned int height)
{
    unsigned int denominator = 0;
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, width);
    out_array[index] = 0;
    for (int dx = -radius; dx <= radius; ++dx)
    {
        for (int dy = -radius; dy <= radius; ++dy)
        {
            const int neighborX = x + dx;
            const int neighborY = y + dy;
            if (neighborX < 0 || neighborX >= width || neighborY < 0 || neighborY >= height)
            {
                continue;
            }
            const int neighborIndex = GetPixel(neighborX, neighborY, width);
            out_array[index] += in_array[neighborIndex];
            ++denominator;
        }
    }
    out_array[index] /= denominator;
}
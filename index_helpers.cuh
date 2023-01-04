__device__ unsigned int GetPixelWithPadding(unsigned int x, unsigned int y, unsigned int width, unsigned int padding)
{
    return (x + padding + (y + padding) * (width + 2 * padding)) * gridDim.z;
}

__device__ unsigned int GetPixel(unsigned int x, unsigned int y, unsigned int width)
{
    return GetPixelWithPadding(x, y, width, 0);
}
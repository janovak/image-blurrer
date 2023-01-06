#include <algorithm>
#include <cmath>
#include <stdio.h>
#include "index_helpers.cuh"

#define FILTER_RADIUS 1
#define FILTER_WIDTH (FILTER_RADIUS * 2 + 1)

__device__ void WarpReduce(volatile int16_t* minData, volatile int16_t* maxData, unsigned int tid)
{
    if (blockDim.x >= 64)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 32]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 32]);
    }
    if (blockDim.x >= 32)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 16]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 16]);
    }
    if (blockDim.x >= 16)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 8]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 8]);
    }
    if (blockDim.x >= 8)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 4]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 4]);
    }
    if (blockDim.x >= 4)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 2]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 2]);
    }
    if (blockDim.x >= 2)
    {
        minData[tid] = std::min(minData[tid], minData[tid + 1]);
        maxData[tid] = std::max(maxData[tid], maxData[tid + 1]);
    }
}

extern "C" __global__ void GetMinMax(int16_t *in_array, int16_t *out_minArray, int16_t *out_maxArray, unsigned int length)
{
    extern __shared__ int16_t shared_minArray[];
    extern __shared__ int16_t shared_maxArray[];
    const unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if (i > length)
    {
        return;
    }
    shared_minArray[tid] = in_array[i];
    shared_maxArray[tid] = in_array[i];
    while (i < length)
    {
        shared_minArray[tid] = std::min(shared_minArray[i], in_array[i + blockDim.x]);
        shared_maxArray[tid] = std::max(shared_maxArray[i], in_array[i + blockDim.x]);
        i += gridDim.x;
    }
    __syncthreads();
    if (blockDim.x >= 512)
    {
        if (tid < 256)
        {
            shared_minArray[tid] = std::min(shared_minArray[tid], shared_minArray[tid + 256]);
            shared_maxArray[tid] = std::min(shared_maxArray[tid], shared_maxArray[tid + 256]);
        }
        __syncthreads();
    }
    if (blockDim.x >= 256)
    {
        if (tid < 128)
        {
            shared_minArray[tid] = std::min(shared_minArray[tid], shared_minArray[tid + 128]);
            shared_maxArray[tid] = std::min(shared_maxArray[tid], shared_maxArray[tid + 128]);
        }
        __syncthreads();
    }
    if (blockDim.x >= 128)
    {
        if (tid < 64)
        {
            shared_minArray[tid] = std::min(shared_minArray[tid], shared_minArray[tid + 64]);
            shared_maxArray[tid] = std::min(shared_maxArray[tid], shared_maxArray[tid + 64]);
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        WarpReduce(shared_minArray, shared_maxArray, tid);
    }
    if (tid == 0)
    {
        out_minArray[blockIdx.x] = shared_minArray[0];
        out_maxArray[blockIdx.x] = shared_maxArray[0];
    }
}

extern "C" __global__ void SobelFilter(uint8_t *in_array, int16_t *out_array, int16_t *filterArray, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, width) + blockIdx.z;
    int sum = 0;
#pragma unroll
    for (int i = -FILTER_RADIUS; i <= FILTER_RADIUS; ++i)
    {
        const int neighborY = y + i;
#pragma unroll
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; ++j)
        {
            const int neighborX = x + j;
            const unsigned int neighborIndex = GetPixelWithPadding(neighborX, neighborY, width, FILTER_RADIUS) + blockIdx.z;
            sum += in_array[neighborIndex] * filterArray[(i + FILTER_RADIUS) * FILTER_WIDTH + j + FILTER_RADIUS];
        }
    }
    out_array[index] = sum;
}

extern "C" __global__ void EdgeDetection(int16_t *in_xSobelFilter, int16_t *in_ySobelFilter, uint8_t *out_array, unsigned int length, int min, int max)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y * gridDim.x + x > length)
    {
        return;
    }
    const unsigned int index = GetPixel(x, y, length) + blockIdx.z;
    const float sobelValue = sqrt(pow(in_xSobelFilter[index], 2) + pow(in_ySobelFilter[index], 2));
    out_array[index] = (sobelValue - min) / max * 255;
}
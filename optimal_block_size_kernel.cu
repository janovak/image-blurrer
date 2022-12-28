 #include <cmath>

 __constant__ const static unsigned int BLOCK_SIZE = 128;
 // Every two indices fors a pair of factors that multiply to a multiple of 32 between 128 and 256.
 // As these are used to minimize the waste of unused threads, INT_MAX is used for the extra elements so they naturally won't be picked for a minimum
 __constant__ const static unsigned int BLOCK_DIMENSIONS[BLOCK_SIZE] = {1, 128, 2, 64, 4, 32, 8, 16, 16, 8, 32, 4, 64, 2, 128, 1,
                                                                        1, 160, 2, 80, 4, 40, 5, 32, 8, 20, 10, 16, 16, 10, 20, 8, 32, 5, 40, 4, 80, 2, 160, 1,
                                                                        1, 192, 2, 96, 3, 64, 4, 48, 6, 32, 8, 24, 12, 16, 16, 12, 24, 8, 32, 6, 48, 4, 64, 3, 96, 2, 192, 1,
                                                                        1, 224, 2, 112, 4, 56, 7, 32, 8, 28, 14, 16, 16, 14, 28, 8, 32, 7, 56, 4, 112, 2, 224, 1,
                                                                        1, 256, 2, 128, 4, 64, 8, 32, 16, 16, 32, 8, 64, 4, 128, 2, 256, 1,
                                                                        INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX };

 struct Block
 {
    unsigned int waste;
    unsigned int w;
    unsigned int h;

    __device__ void operator=(volatile Block& rhs) volatile
    {
       waste = rhs.waste;
       w = rhs.w;
       h = rhs.h;
    }

    __device__ Block()
    {
    }

    __device__ Block(unsigned int waste, unsigned int w, unsigned int h)
        : waste(waste), w(w), h(h)
    {
    }
 };

 __device__ void WarpReduce(volatile Block *data, unsigned int tid)
 {
    if (data[tid].waste > data[tid + 32].waste) data[tid] = data[tid + 32];
    if (data[tid].waste > data[tid + 16].waste) data[tid] = data[tid + 16];
    if (data[tid].waste > data[tid + 8].waste) data[tid] = data[tid + 8];
    if (data[tid].waste > data[tid + 4].waste) data[tid] = data[tid + 4];
    if (data[tid].waste > data[tid + 2].waste) data[tid] = data[tid + 2];
    if (data[tid].waste > data[tid + 1].waste) data[tid] = data[tid + 1];
 }

extern "C" __global__ void FindOptimalBlockDimensions(unsigned int width, unsigned int height, unsigned int *optimalDimensions)
{
    __shared__ Block waste[64];
    const unsigned int tid = threadIdx.x;
    const unsigned int index = tid * 2;
    const unsigned int w = BLOCK_DIMENSIONS[index];
    const unsigned int h = BLOCK_DIMENSIONS[index + 1];
    // For a given block size (w x h), calculate how many blocks it would take to cover the dimensions being passed in.
    // Store the difference (i.e. waste) so we can minimize it next.
    waste[tid] = Block(static_cast<unsigned int>(ceil(float(width) / w) * w * ceil(float(height) / h) * h - width * height), w, h);

    __syncthreads();

    if (tid < 32)
    {
        WarpReduce(waste, tid);
    }
    if (tid == 0)
    {
        // Return the optimal block dimensions
        optimalDimensions[0] = waste[0].w;
        optimalDimensions[1] = waste[0].h;
    }
}
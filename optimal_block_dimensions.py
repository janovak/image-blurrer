import math
import numpy
import pycuda.autoinit
import pycuda.driver as cuda
import sys

def get_optimal_block_size_serial(image_width, image_height):
    factors = {
        128 : [1, 2, 4, 8, 16, 32, 64, 128],
        160 : [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160],
        192 : [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 192],
        224 : [1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112, 224],
        256 : [1, 2, 4, 8, 16, 32, 64, 128, 256]
    }

    min_w = 0
    min_h = 0
    min_waste = sys.maxsize
    image_area = image_width * image_height
    # Iterate over all pairs of factors that multiply to a multiple of 32 between 128 and 256
    for block_size in range(128, 257, 32):
        for w in factors[block_size]:
            h = block_size / w
            # Calculate how many blocks of size w x h would be needed to cover the image and how much waste there would be
            waste = math.ceil(image_width / w) * w * math.ceil(image_height / h) * h - image_area
            if min_waste > waste:
                min_waste = waste
                min_w = w
                min_h = h
    return min_w, min_h

def get_optimal_block_size_parallel(image_width, image_height):
    # Set up the arrays for the device
    result = numpy.empty(2, dtype=numpy.int32)
    result_gpu = cuda.mem_alloc(result.nbytes)
    cuda.memcpy_htod(result_gpu, result)

    # Get the function from the kernel
    mod = cuda.module_from_file('optimal_block_size_kernel.cubin')
    optimal_block_dimensions = mod.get_function('FindOptimalBlockDimensions')

    # Calculate the optimal block dimensions on the device
    optimal_block_dimensions(numpy.int32(image_width), numpy.int32(image_height), result_gpu, block=(64, 1, 1), grid=(1, 1, 1))
    cuda.memcpy_dtoh(result, result_gpu)

    return result
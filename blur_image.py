import math
import numpy
import imageio.v2 as imageio
import sys
import optimal_block_dimensions as block_dims
import os
import pycuda.autoinit
import pycuda.driver as cuda
from typing import Final

PIXEL_ATTRIBUTES: Final[int] = 3

path, blur_level = sys.argv[1], int(sys.argv[2])
# Open the image
original_img = imageio.imread(path, pilmode='RGB')
# Break the file path into its name and extension
filename = os.path.splitext(path)
name = os.path.basename(filename[0])
ext = filename[1]
image_width = len(original_img[0])
image_height = len(original_img)

# Calculate the block dimensions to use
block_width, block_height = block_dims.get_optimal_block_dimensions_parallel(image_width, image_height)
grid_width = math.ceil(image_width / block_width)
grid_height = math.ceil(image_height / block_height)

# Flatten the image into a 1D array
flattened_img = original_img.flatten().astype(numpy.int32)
flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
# Create an empty result array
result = numpy.empty(len(flattened_img), dtype=numpy.int32)
result_gpu = cuda.mem_alloc(result.nbytes)
# Copy input and output arrays to the device
cuda.memcpy_htod(flattened_img_gpu, flattened_img)
cuda.memcpy_htod(result_gpu, result)

# Call the method on the device to blur the image
mod = cuda.module_from_file('blur_image_kernel.cubin')
box_blur = mod.get_function('BoxBlur')
box_blur(flattened_img_gpu,
            result_gpu,
            numpy.int32(blur_level),
            numpy.int32(image_width),
            numpy.int32(image_height),
            block=(block_width, block_height, 1),
            grid=(grid_width, grid_height, PIXEL_ATTRIBUTES))
cuda.memcpy_dtoh(result, result_gpu)

# Write the blurred image to disk
imageio.imwrite(name + '_blurred' + str(blur_level) + ext, numpy.reshape(result.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
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
BLUR_RADIUS: Final[int] = 25

path = sys.argv[1]
# Open the image
original_img = imageio.imread(path, pilmode='RGB')
# Break the file path into its name and extension
filepath = os.path.splitext(path)
image_width = len(original_img[0])
image_height = len(original_img)
# Calculate the block dimensions to use
block_width, block_height = block_dims.get_optimal_block_dimensions_parallel(image_width, image_height)
grid_width = math.ceil(image_width / block_width)
grid_height = math.ceil(image_height / block_height)

# Pad and flatten the image into a 1D array
padded_img = numpy.pad(original_img, [(BLUR_RADIUS, BLUR_RADIUS), (BLUR_RADIUS, BLUR_RADIUS), (0, 0)], mode='edge')
flattened_img = padded_img.flatten().astype(numpy.uint8)
flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
# Create an empty result array
result = numpy.empty(original_img.size, dtype=numpy.uint8)
result_gpu = cuda.mem_alloc(result.nbytes)
# Copy input and output arrays to the device
cuda.memcpy_htod(flattened_img_gpu, flattened_img)
cuda.memcpy_htod(result_gpu, result)
# Call the method on the device to blur the image
mod = cuda.module_from_file('blur_image_kernel.cubin')
"""
box_blur = mod.get_function('BoxBlur')
box_blur(flattened_img_gpu,
            result_gpu,
            numpy.int32(image_width),
            numpy.int32(image_height),
            block=(block_width, block_height, 1),
            grid=(grid_width, grid_height, PIXEL_ATTRIBUTES))
"""
kernel_diameter = BLUR_RADIUS * 2 + 1
sigma_squared = pow(BLUR_RADIUS / 2, 2)
gaussian_kernel = numpy.empty([kernel_diameter, kernel_diameter], dtype=numpy.float32)
for i in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
    for j in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
        exponent_numerator = i * i + j * j
        two_sigma_squared = 2 * sigma_squared
        gaussian_kernel[i + BLUR_RADIUS][j + BLUR_RADIUS] = math.exp(-exponent_numerator / two_sigma_squared) / (two_sigma_squared * math.pi)
gaussian_kernel /= gaussian_kernel.sum()
gaussian_kernel.flatten().astype(numpy.float32)
gaussian_kernel_gpu = cuda.mem_alloc(gaussian_kernel.nbytes)
# Copy input and output arrays to the device
cuda.memcpy_htod(gaussian_kernel_gpu, gaussian_kernel)
gaussian_blur = mod.get_function('GaussianBlur')
gaussian_blur(flattened_img_gpu,
            result_gpu,
            gaussian_kernel_gpu,
            numpy.int32(image_width),
            numpy.int32(image_height),
            block=(block_width, block_height, 1),
            grid=(grid_width, grid_height, PIXEL_ATTRIBUTES))
cuda.memcpy_dtoh(result, result_gpu)

# Write the blurred image to disk
imageio.imwrite(filepath[0] + '_blurred' + str(BLUR_RADIUS) + filepath[1], numpy.reshape(result.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
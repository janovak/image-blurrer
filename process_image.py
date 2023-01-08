import math
import numpy
import imageio.v2 as imageio
import sys
import optimal_block_dimensions as block_dims
import os
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from typing import Final

PIXEL_ATTRIBUTES: Final[int] = 3
BLUR_RADIUS: Final[int] = 1
SOBEL_FILTER_RADIUS: Final[int] = 1

class Dimensions:
    def __init__(self, image_width, image_height, block_width, block_height, grid_width, grid_height):
        self.image_width = image_width
        self.image_height = image_height
        self.block_width = block_width
        self.block_height = block_height
        self.grid_width = grid_width
        self.grid_height = grid_height

def get_gaussian_kernel():
    kernel_diameter = BLUR_RADIUS * 2 + 1
    two_sigma_squared = 2 * pow(BLUR_RADIUS / 2, 2)
    gaussian_kernel = numpy.empty([kernel_diameter, kernel_diameter], dtype=numpy.float32)
    for i in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
        for j in range(-BLUR_RADIUS, BLUR_RADIUS + 1):
            exponent_numerator = i * i + j * j
            gaussian_kernel[i + BLUR_RADIUS][j + BLUR_RADIUS] = math.exp(-exponent_numerator / two_sigma_squared) / (two_sigma_squared * math.pi)
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel.flatten().astype(numpy.float32)
    return gaussian_kernel

def box_blur(dimensions, original_img):
    # Pad and flatten the image into a 1D array
    padded_img = numpy.pad(original_img, [(BLUR_RADIUS, BLUR_RADIUS), (BLUR_RADIUS, BLUR_RADIUS), (0, 0)], mode='edge')
    flattened_img_gpu = gpuarray.to_gpu(padded_img.flatten().astype(numpy.uint8))
    # Create an empty result array
    blurred_img_gpu = gpuarray.to_gpu(numpy.empty(original_img.size, dtype=numpy.uint8))
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('blur_image_kernel.cubin')
    box_blur = mod.get_function('BoxBlur')
    box_blur(flattened_img_gpu,
                blurred_img_gpu,
                numpy.int32(dimensions.image_width),
                numpy.int32(dimensions.image_height),
                block=(dimensions.block_width, dimensions.block_height, 1),
                grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    return blurred_img_gpu.get()

def gaussian_blur(dimensions, original_img):
    # Pad and flatten the image into a 1D array
    padded_img = numpy.pad(original_img, [(BLUR_RADIUS, BLUR_RADIUS), (BLUR_RADIUS, BLUR_RADIUS), (0, 0)], mode='edge')
    flattened_img_gpu = gpuarray.to_gpu(padded_img.flatten().astype(numpy.uint8))
    # Create an empty result array
    blurred_img_gpu = gpuarray.to_gpu(numpy.empty(original_img.size, dtype=numpy.uint8))
    # Generate the gaussian kernel
    gaussian_kernel_gpu = gpuarray.to_gpu(get_gaussian_kernel())
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('blur_image_kernel.cubin')
    gaussian_blur = mod.get_function('GaussianBlur')
    gaussian_blur(flattened_img_gpu,
                    blurred_img_gpu,
                    gaussian_kernel_gpu,
                    numpy.int32(dimensions.image_width),
                    numpy.int32(dimensions.image_height),
                    block=(dimensions.block_width, dimensions.block_height, 1),
                    grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    return blurred_img_gpu.get()

def grayscale_filter(dimensions, original_img):
    # Flatten the image into a 1D array
    flattened_img_gpu = gpuarray.to_gpu(original_img.flatten().astype(numpy.uint8))
    # Create an empty result array
    result_gpu = gpuarray.to_gpu(numpy.empty(original_img.size, dtype=numpy.uint8))
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('grayscale_kernel.cubin')
    box_blur = mod.get_function('GrayscaleFilter')
    box_blur(flattened_img_gpu,
                result_gpu,
                numpy.int32(dimensions.image_width),
                numpy.int32(dimensions.image_height),
                block=(dimensions.block_width, dimensions.block_height, 1),
                grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    return result_gpu.get()

def sobel_filter(dimensions, original_img):
    grayscaled_img = grayscale_filter(dimensions, original_img)
    blurred_and_grayed_img = gaussian_blur(dimensions, numpy.reshape(grayscaled_img.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
    preprocessed_img = numpy.pad(numpy.reshape(blurred_and_grayed_img.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)), [(SOBEL_FILTER_RADIUS, SOBEL_FILTER_RADIUS), (SOBEL_FILTER_RADIUS, SOBEL_FILTER_RADIUS), (0, 0)], mode='edge')
    preprocessed_img_gpu =  gpuarray.to_gpu(preprocessed_img.flatten())
    # Create an empty filtered result array
    filtered_img_gpu = gpuarray.to_gpu(numpy.empty(original_img.size, dtype=numpy.int16))
    # Call the method on the device to get the min and max values in the matrix
    mod = cuda.module_from_file('sobel_kernel.cubin')
    sobel_filter = mod.get_function('SobelFilter')
    sobel_filter(preprocessed_img_gpu,
                    filtered_img_gpu,
                    numpy.int32(dimensions.image_width),
                    numpy.int32(dimensions.image_height),
                    block=(dimensions.block_width, dimensions.block_height, 1),
                    grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    # Get min and max values in the filtered array
    max = gpuarray.max(filtered_img_gpu)
    min = gpuarray.min(filtered_img_gpu)
    # Create an empty result array
    result_gpu = gpuarray.to_gpu(numpy.empty(original_img.size, dtype=numpy.uint8))
    # Call the method on the device to the values in the matrix
    normalize_colors = mod.get_function('NormalizeColors')
    normalize_colors(filtered_img_gpu,
                        result_gpu,
                        numpy.int32(dimensions.image_width),
                        numpy.int32(dimensions.image_height),
                        min,
                        max,
                        block=(dimensions.block_width, dimensions.block_height, 1),
                        grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    return result_gpu.get()

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
dimensions = Dimensions(image_width, image_height, block_width, block_height, grid_width, grid_height)

# Calculate and write the blurred image to disk
box_blurred_image = box_blur(dimensions, original_img)
imageio.imwrite(filepath[0] + '_boxblurred' + str(BLUR_RADIUS) + filepath[1], numpy.reshape(box_blurred_image.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))

# Calculate and write the blurred image to disk
gaussian_blurred_image = gaussian_blur(dimensions, original_img)
imageio.imwrite(filepath[0] + '_gaussianblurred' + str(BLUR_RADIUS) + filepath[1], numpy.reshape(gaussian_blurred_image.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))

# Calculate and write the blurred image to disk
grayscaled_image = grayscale_filter(dimensions, original_img)
imageio.imwrite(filepath[0] + '_grayscaled' + filepath[1], numpy.reshape(grayscaled_image.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))

# Calculate and write the Sobel filter to disk
sobel_filtered_image = sobel_filter(dimensions, original_img)
imageio.imwrite(filepath[0] + '_sobelfiltered' + filepath[1], numpy.reshape(sobel_filtered_image.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
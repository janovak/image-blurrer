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
BLUR_RADIUS: Final[int] = 1

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
    flattened_img = padded_img.flatten().astype(numpy.uint8)
    flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
    # Create an empty result array
    blurred_img = numpy.empty(original_img.size, dtype=numpy.uint8)
    blurred_img_gpu = cuda.mem_alloc(blurred_img.nbytes)
    # Copy input and output arrays to the device
    cuda.memcpy_htod(flattened_img_gpu, flattened_img)
    cuda.memcpy_htod(blurred_img_gpu, blurred_img)
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('blur_image_kernel.cubin')
    box_blur = mod.get_function('BoxBlur')
    box_blur(flattened_img_gpu,
                blurred_img_gpu,
                numpy.int32(dimensions.image_width),
                numpy.int32(dimensions.image_height),
                block=(dimensions.block_width, dimensions.block_height, 1),
                grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    cuda.memcpy_dtoh(blurred_img, blurred_img_gpu)
    return blurred_img

def gaussian_blur(dimensions, original_img):
    # Pad and flatten the image into a 1D array
    padded_img = numpy.pad(original_img, [(BLUR_RADIUS, BLUR_RADIUS), (BLUR_RADIUS, BLUR_RADIUS), (0, 0)], mode='edge')
    flattened_img = padded_img.flatten().astype(numpy.uint8)
    flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
    # Create an empty result array
    blurred_img = numpy.empty(original_img.size, dtype=numpy.uint8)
    blurred_img_gpu = cuda.mem_alloc(blurred_img.nbytes)
    # Copy input and output arrays to the device
    cuda.memcpy_htod(flattened_img_gpu, flattened_img)
    cuda.memcpy_htod(blurred_img_gpu, blurred_img)
    # Generate the gaussian kernel
    gaussian_kernel = get_gaussian_kernel()
    gaussian_kernel_gpu = cuda.mem_alloc(gaussian_kernel.nbytes)
    cuda.memcpy_htod(gaussian_kernel_gpu, gaussian_kernel)
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
    cuda.memcpy_dtoh(blurred_img, blurred_img_gpu)
    return blurred_img

def grayscale_filter(dimensions, original_img):
    # Flatten the image into a 1D array
    flattened_img = original_img.flatten().astype(numpy.uint8)
    flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
    # Create an empty result array
    result = numpy.empty(original_img.size, dtype=numpy.uint8)
    result_gpu = cuda.mem_alloc(result.nbytes)
    # Copy input and output arrays to the device
    cuda.memcpy_htod(flattened_img_gpu, flattened_img)
    cuda.memcpy_htod(result_gpu, result)
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('grayscale_kernel.cubin')
    box_blur = mod.get_function('GrayscaleFilter')
    box_blur(flattened_img_gpu,
                result_gpu,
                numpy.int32(dimensions.image_width),
                numpy.int32(dimensions.image_height),
                block=(dimensions.block_width, dimensions.block_height, 1),
                grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    cuda.memcpy_dtoh(result, result_gpu)
    return result

def sobel_filter_one_dimension(dimensions, original_img, filter_array_gpu):   
    # Pad and flatten the image into a 1D array
    padded_img = numpy.pad(original_img, [(1, 1), (1, 1), (0, 0)], mode='edge')
    flattened_img = padded_img.flatten().astype(numpy.uint8)
    flattened_img_gpu = cuda.mem_alloc(flattened_img.nbytes)
    # Create an empty result array
    x_sobel_filtered_img = numpy.empty(original_img.size, dtype=numpy.int16)
    x_sobel_filtered_img_gpu = cuda.mem_alloc(x_sobel_filtered_img.nbytes)
    # Copy input and output arrays to the device
    cuda.memcpy_htod(flattened_img_gpu, flattened_img)
    cuda.memcpy_htod(x_sobel_filtered_img_gpu, x_sobel_filtered_img)
    # Call the method on the device to blur the image
    mod = cuda.module_from_file('sobel_kernel.cubin')
    sobel_filter = mod.get_function('SobelFilter')
    sobel_filter(flattened_img_gpu,
                    x_sobel_filtered_img_gpu,
                    filter_array_gpu,
                    numpy.int32(dimensions.image_width),
                    numpy.int32(dimensions.image_height),
                    block=(dimensions.block_width, dimensions.block_height, 1),
                    grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))

    # Create an empty result array
    min_array = numpy.empty(original_img.size, dtype=numpy.int16)
    min_array_gpu = cuda.mem_alloc(min_array.nbytes)
    # Create an empty result array
    max_array = numpy.empty(original_img.size, dtype=numpy.int16)
    max_array_gpu = cuda.mem_alloc(max_array.nbytes)
    # Copy output arrays to the device
    cuda.memcpy_htod(min_array_gpu, min_array)
    cuda.memcpy_htod(max_array_gpu, max_array)
    # Call the method on the device to get the min and max values in the matrix
    get_min_max = mod.get_function('GetMinMax')
    get_min_max(x_sobel_filtered_img_gpu,
                min_array_gpu,
                max_array_gpu,
                numpy.int32(dimensions.image_width * dimensions.image_height),
                block=(dimensions.block_width, dimensions.block_height, 1),
                grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    cuda.memcpy_dtoh(min_array, min_array_gpu)
    cuda.memcpy_dtoh(max_array, max_array_gpu)
    
    return x_sobel_filtered_img_gpu, min_array[0], max_array[0]

def sobel_filter(dimensions, original_img):
    grayscaled_img = grayscale_filter(dimensions, original_img)
    blurred_and_grayed_img = gaussian_blur(dimensions, numpy.reshape(grayscaled_img.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
    preprocessed_img = numpy.reshape(blurred_and_grayed_img.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES))

    x_filter_array = numpy.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=numpy.int16).flatten()
    x_filter_array_gpu = cuda.mem_alloc(x_filter_array.nbytes)
    cuda.memcpy_htod(x_filter_array_gpu, x_filter_array)
    x_sobel_filter_img, x_min, x_max = sobel_filter_one_dimension(dimensions, preprocessed_img, x_filter_array_gpu)

    y_filter_array = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=numpy.int16).flatten()
    y_filter_array_gpu = cuda.mem_alloc(y_filter_array.nbytes)
    cuda.memcpy_htod(y_filter_array_gpu, y_filter_array)
    y_sobel_filter_img, y_min, y_max = sobel_filter_one_dimension(dimensions, preprocessed_img, y_filter_array_gpu)
    
    # Create an empty result array
    result = numpy.empty(original_img.size, dtype=numpy.uint8)
    result_gpu = cuda.mem_alloc(result.nbytes)
    cuda.memcpy_htod(result_gpu, result)

    # Call the method on the device to get the min and max values in the matrix
    mod = cuda.module_from_file('sobel_kernel.cubin')
    edge_detection = mod.get_function('EdgeDetection')
    edge_detection(x_sobel_filter_img,
                    y_sobel_filter_img,
                    result_gpu,
                    numpy.int32(dimensions.image_width * dimensions.image_height),
                    min(x_min, y_min),
                    max(x_max, y_max),
                    block=(dimensions.block_width, dimensions.block_height, 1),
                    grid=(dimensions.grid_width, dimensions.grid_height, PIXEL_ATTRIBUTES))
    cuda.memcpy_dtoh(result, result_gpu)

    return result

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

# Calculate and write the blurred image to disk
sobel_filtered_image = sobel_filter(dimensions, original_img)
imageio.imwrite(filepath[0] + '_sobelfiltered' + filepath[1], numpy.reshape(sobel_filtered_image.astype(numpy.uint8), (image_height, image_width, PIXEL_ATTRIBUTES)))
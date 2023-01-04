# image-processing

Provides PyCuda implementations of varius image processing algorithms.

The CUDA kernel files need to be compiled with `nvcc --cubin <file>.cu` before the program will work.

Generates outputs based on the input file's name with additions to indicate which output is from which algorithm.

For example</br>
Input: `picture.png`</br>
Output: `picture_boxblurred25.png` and `picture_gaussianblurred25.png` and `picture_grayscaled.png`

## Image blurring
Blurs an image using the box blur and Gaussian blur algorithm.

Blur radius is not currently configurable without a recompile. Changing the blur radius requires changing the `BLUR_RADIUS` variables at the top of both `process_image.py` and `blur_image_kernel.cu`

## Grayscale filter
Applies a grayscale filter to a given image.
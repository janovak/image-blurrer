# image-processing

Provides PyCuda implementations of varius image processing algorithms.

## Image blurring
Blurs an image using the box blur and Gaussian blur algorithm.

Blur radius is not currently configurable without a recompile. Changing the blur radius requires changing the `BLUR_RADIUS` variables at the top of both `process_image.py` and `blur_image_kernel.cu`

Generates two outputs with the names
</br>
`<input file name>_boxblurred<blur radius>.<input extension>` and `<input file name>_gaussianblurred<blur radius>.<input extension>`
</br>
e.g.
</br>
Input: `picture.png`
</br>
Output: `picture_boxblurred25.png` and `picture_gaussianblurred25.png`
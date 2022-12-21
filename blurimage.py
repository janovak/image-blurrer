import numpy
import imageio.v2 as imageio
import sys
import os
from numpy import mean

# returns a list of all the items in the 2D array within the specified distance of the given point
def neighbors(matrix, radius, row, column):
    return [matrix[i][j]
                for j in range(max(0, column - radius), min(column + radius + 1, len(matrix[0])))
                    for i in range(max(0, row - radius), min(row + radius + 1, len(matrix)))]

# iterates over the pixels in the image and blurs them using the box blur algorithm
# blur_level is the radius of the box blur algorithm
def blur_image(img, blur_level):
    output_img = numpy.empty((len(img), len(img[0]), 3))
    for r in range(0, len(img)):
        for c in range(0, len(img[0])):
            n = neighbors(img, blur_level, r, c)
            output_img[r][c] = tuple(mean(n, axis=0))
    return output_img

# get image name and amount to blur the picture from the command line arguments
path, blur_level = sys.argv[1], int(sys.argv[2])
# open the image
img = imageio.imread(path, pilmode='RGB')
# break the file path into its name and extension
filename = os.path.splitext(path)
name = os.path.basename(filename[0])
ext = filename[1]
# write the blurred image to disk
imageio.imwrite(name + '_blurred' + str(blur_level) + ext, blur_image(img, blur_level))
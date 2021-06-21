import numpy as np
from math import sqrt, pi, cos, sin
import matplotlib.pyplot as plt
from collections import defaultdict
from skimage.io import imread
import matplotlib
from skimage import feature
from skimage.color import *
from collections import defaultdict
from skimage.draw import circle_perimeter
import skimage

# reading the image
image = imread("ball-bearings-1958085_1280.jpg")


# defining the function to find circle in image
def CountShapes_ID(image):
    # gray conversion of image
    gray = rgb2gray(image)
    # detecting edges of the image
    edge_image = feature.canny(gray, sigma=2, low_threshold=0.1, high_threshold=0.5)
    # minimum radius
    rmin = 10
    # maximum radius
    rmax = 60
    # steps to iterate over from 0 to 2pi
    theta = 100

    listOfCoordinates = []
    for radius in range(rmin, rmax + 1):
        for step in range(theta):
            equation = (radius, int(radius*cos(2*pi*step/theta)), int(radius*sin(2*pi*step/theta)))
            listOfCoordinates.append(equation)

    height, width = edge_image.shape

    centre = defaultdict(int)
    for x in range(height):
        for y in range(width):
            # skip for weak pixels
            if edge_image[x][y] != 0:
                for r, cost, sint in listOfCoordinates:
                    # getting centre of the circle
                    a = x - cost
                    b = y - sint
                    centre[(a, b, r)] += 1
    # setting threshold to match the circle
    threshold = 0.4
    possibleCircles = []

    for circle, count in centre.items():
        x, y, r = circle
        # determining the threshold
        if count / theta >= threshold:
            possibleCircles.append((x, y, r))

    M, N = edge_image.shape
    # empty image
    img = np.zeros((M, N), dtype=np.uint8)

    for x, y, r in possibleCircles:
        # creating the perimeter of circle
        rr, cc = skimage.draw.circle_perimeter(x, y, r, method='bresenham', shape=None)
        # setting value of coordinates to 1 with radius r
        img[rr, cc] = 1

    return img

shape = CountShapes_ID(image)

matplotlib.image.imsave('circles.jpg', shape)
fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(shape)
ax[1].set_title('Circles identified')
ax[1].axis('off')
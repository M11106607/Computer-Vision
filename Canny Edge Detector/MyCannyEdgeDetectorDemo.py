from skimage import feature
from skimage.metrics import structural_similarity
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import *
import numpy as np
from scipy.ndimage.filters import convolve

plt.gray()

def myCannyEdgeDetector(image, Low_Threshold, High_Threshold):
    width, height, depth = image.shape
    grayImage = rgb2gray(image)

    # result = gaussian_filter(grayImage, sigma=5)

    # Gaussian Filter
    kernel = 11
    sigma = 2
    kernel = int(kernel)//2
    x, y = np.mgrid[-kernel:kernel + 1, -kernel:kernel + 1]
    div = 1/np.multiply(2,np.pi)*np.square(sigma)
    gaus = np.exp(-((np.square(x)+ np.square(y))/(np.multiply(2,np.square(sigma)))))*div

    result = convolve(grayImage/255,gaus)

# Prewitt filter

# x = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
# y = np.array([[1,0,-1],[1,0,-1],[-1,-1,-1]])

    # Sobel filter
    Sobelx = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], np.float64)
    Sobely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float64)


    dx = convolve(result,Sobelx)
    dy = convolve(result,Sobely)

    magnitude = np.sqrt(np.square(dx),np.square(dy))


    # gradient orientation, that is tan inverse
    theta = np.arctan(dy,dx)

    # Non Maximum Suppression
    width1, height1 = magnitude.shape
    suppression = np.zeros((width1, height1),np.float64)
    degree = np.multiply(theta, np.divide(180,np.pi))
    degree[degree < 0] += 180

    for i in range(1, width1 - 1):
        for j in range(1, height1 - 1):

            if (0 <= degree[i, j] <= 180):
                max_sup = max(magnitude[i, j + 1], magnitude[i, j - 1])

            elif (135 <= degree[i, j] <= 315):
                max_sup = max(magnitude[i + 1, j - 1], magnitude[i - 1, j + 1])

            elif (90 <= degree[i, j] <= 270):
                max_sup = max(magnitude[i + 1, j], magnitude[i - 1, j])

            else:
                max_sup = max(magnitude[i - 1, j - 1], magnitude[i + 1, j + 1])

            if (magnitude[i, j] >= max_sup):
                suppression[i, j] = magnitude[i, j]


    # Threshold
    width2, hieght2 = suppression.shape
    threshold = np.zeros((width2, hieght2),np.float64)

    strongPixel_X, strongPixel_Y = np.where(suppression >= suppression.max()*High_Threshold)
    weakPixel_X, weakPixel_Y = np.where((suppression <= suppression.max()*High_Threshold) & (suppression >= Low_Threshold))

    threshold[strongPixel_X, strongPixel_Y] = 255
    threshold[weakPixel_X, weakPixel_Y] = 0

    canny = threshold
    return canny/255

# Reading the image file
image = io.imread('Pickachu.jpg')
# Calling my canny edge detector function
myOutput = myCannyEdgeDetector(image,0.03, 0.08)

# In built canny edge detector
grayImage = rgb2gray(image)
edges2 = feature.canny(grayImage, sigma=2)

# SSIM
ssim = structural_similarity(edges2,myOutput)

# PSNR
def PSNR(inBuilt, myCanny):
    MSE = np.mean(np.square(inBuilt - myCanny))
    pixel = 255
    PSNR = 10*np.log10(np.square(pixel)/np.sqrt(MSE))
    return PSNR,MSE

psnr,mse = PSNR(myOutput,edges2)

print("Structural Similarity Index Metric score: ",ssim)
print("Peak Signal To Noise Ratio score: ",psnr)

fig, ax = plt.subplots(1, 2, figsize=(8,4))
ax[0].imshow(edges2)
ax[0].set_title('In Built Canny, at sigma=3')
ax[0].axis('off')
ax[1].imshow(myOutput)
ax[1].set_title('My Canny, at sigma=3')
ax[1].axis('off')
plt.show(fig)
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
import matplotlib

image = imread('woman-4525714_1280.jpg') #  image-asset.jpeg
# creating the gaussian filter
def gaussian(xy, sigma):
    g =  np.exp(-(xy/(2*sigma**2)))
    return g

# splitting the region for color determination
def quantRegion(pixelRange):
    space = [[0,63], [64,127], [128,191], [192,255]]
    for regionIndex, value in enumerate(space):
        minValue = value[0] # minimum value of range
        maxValue = value[1] # maximum value of range
        # if pixel is in between the space range return the index of pixel
        if pixelRange >= minValue and  pixelRange <= maxValue:
            return regionIndex

def CartoonNizer_ID(image):
    quantImage = np.copy(image)
    empty_list = [[] for _ in range(4)] # creating empty list
    ColorR = ColorG = ColorB = empty_list
    listrange = [0] * 4 # creating list of range 4 as per the space
    RegionR = RegionG = RegionB = listrange

    for index in image:
        for RGB in index:
            pixelR,pixelG,pixelB = RGB[0],RGB[1],RGB[2]
            ColorR[quantRegion(pixelR)].append(pixelR) # getting the index of each region
            ColorG[quantRegion(pixelG)].append(pixelG)
            ColorB[quantRegion(pixelB)].append(pixelB)

    for i in range(4):
        RegionR[i] = np.mean(ColorR[i]) # taking the mean of all values belongs to that region
        RegionG[i] = np.mean(ColorG[i])
        RegionB[i] = np.mean(ColorB[i])
    for x, value in enumerate(image):
        for y, RGB in enumerate(value):
            pixelR,pixelG,pixelB = RGB[0],RGB[1],RGB[2]
            quantImage[x, y][0] = RegionR[quantRegion(pixelR)] # replacing the values of each region by it's mean value
            quantImage[x, y][1] = RegionG[quantRegion(pixelG)]
            quantImage[x, y][2] = RegionB[quantRegion(pixelB)]

    return quantImage

def bilateralFilter(image):
    scale = 10**(-3)
    M, N = image.shape
    Wp = np.zeros((M, N))
    sigmaS = 2
    SigmaR = 0.1
    kernelSize = 2*sigmaS+1
    Iq = image
    Fx = image*scale
    iteration = 5
    while iteration > 0:
        for p in range(-kernelSize, kernelSize + 1):
            for q in range(-kernelSize, kernelSize + 1):
                Ip = np.roll(image, [q, p], axis=[0, 1])
                weight = np.multiply(gaussian(p**2+q**2, sigmaS), gaussian((Ip - Iq)**2, SigmaR)) # gaussian and range filter
                Fx += np.multiply(Ip, weight)
                Wp = Wp + weight
        Iq = Fx / Wp
        iteration -= 1

    return Fx / Wp

R = bilateralFilter(image[:,:,0])
G = bilateralFilter(image[:,:,1])
B = bilateralFilter(image[:,:,2])

BilateralFilter = np.stack([R,G,B], axis=2 ).astype(np.int32) # stackng all the regions together to get RGB image.
cartoonImage = CartoonNizer_ID(BilateralFilter)

matplotlib.image.imsave('Cartoon.jpg', cartoonImage.astype(np.float32)/255.0)
fig, ax = plt.subplots(1, 3, figsize=(8,4))
ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(BilateralFilter)
ax[1].set_title('Bilateral Filter')
ax[1].axis('off')
ax[2].imshow(cartoonImage)
ax[2].set_title('Color Quantization Result')
ax[2].axis('off')
plt.show(fig)
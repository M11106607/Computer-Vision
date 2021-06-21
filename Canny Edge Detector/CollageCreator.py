import skimage
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import *
import numpy as np

def CollageCreate(Collage):

    image_01 = io.imread(Collage+"\image_01.png")
    image_02 = io.imread(Collage+"\image_03.png")
    image_03 = io.imread(Collage+"\image_02.png")
    image_04 = io.imread(Collage+"\image_05.png")
    image_05 = io.imread(Collage+"\image_04.png")
    image_06 = io.imread(Collage+"\image_06.png")

    #Using vstack and hstack to join the images
    collage01 = np.vstack([image_01,image_02])
    collage02 = np.vstack([image_03,image_04])
    collage03 = np.vstack([image_05,image_06])
    final = np.hstack([collage01,collage02,collage03])
    withoutOverlay = np.hstack([collage01,collage02,collage03])

    # taking matrix of area to be overlay
    mat_1 = final[:, 1120:1280]
    mat_2 = final[:, 1281:1441]
    result = np.add(mat_1, mat_2)
    final[:, 1120:1280] = result
    # final[:, 1120:1280]
    mat_3 = final[:, 2400:2560]
    mat_4 = final[:, 2561:2721]
    result1 = np.add(mat_3, mat_4)
    final[:, 2400:2560] = result1
    # overlaying the images
    end = np.hstack([final[:, :1280], final[:, 1441:2560], final[:, 2721:]])
    return end, withoutOverlay


path = '/photos'
withOverlay, withoutOverlay = CollageCreate(path)

fig, ax = plt.subplots(2, 1, figsize=(18,8))
ax[0].imshow(withoutOverlay)
ax[0].set_title('Collage Without Overlay')
ax[0].axis('off')
ax[1].imshow(withOverlay)
ax[1].set_title('Collage With Overlay')
ax[1].axis('off')
plt.show(fig)

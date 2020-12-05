# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:30:29 2020

@author: toml
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import exposure

img = cv.imread('washed_out_pollen_image.jpg', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]

plt.hist(img.ravel(),150,[85,140])
plt.title('Orignal image histogram')
plt.show()

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))


fig = plt.figure(figsize=(12, 10))

fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Orignal Image')
fig.add_subplot(1, 2, 2)
plt.imshow(img_rescale, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Image After contrast stretching')

plt.show()

plt.hist(img_rescale.ravel(),150,[0,256])
plt.title('histogram after contrast stretching ')
plt.show()

equ = cv.equalizeHist(img)
plt.imshow(np.hstack((img,img_rescale,equ)), cmap='gray')
# plt.title('Orignal image, contrast stretching and the image after equalization')
plt.xticks([]), plt.yticks([])

# fig = plt.figure(figsize=(12, 10))

# fig.add_subplot(1, 2, 1)
# plt.imshow(img_rescale, cmap = 'gray')
# plt.xticks([]), plt.yticks([])
# plt.title('Image After contrast stretching')
# fig.add_subplot(1, 2, 2)
# plt.imshow(equ, cmap = 'gray')
# plt.xticks([]), plt.yticks([])
# plt.title('Image After histogram equalization ')
# plt.show()

# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 12:30:29 2020

@author: toml
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# from scipy.signal import convolve2d as conv2
# from numpy.fft import fft2, ifft2, ifftshift,fftshift
# from skimage import restoration
# from scipy import ndimage
# from numpy import sum,sqrt
# from numpy.random import standard_normal
# from skimage.util import random_noise 
from skimage import exposure

img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]

# hist = cv2.calcHist([img],[0],None,[256],[0,256])

plt.hist(img.ravel(),50,[0,256])
plt.show()

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))


fig = plt.figure(figsize=(12, 10))

fig.add_subplot(1, 2, 1)
plt.imshow(img, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('Image')
fig.add_subplot(1, 2, 2)
plt.imshow(img_rescale, cmap = 'gray')
plt.xticks([]), plt.yticks([])
plt.title('After contrast')

plt.show()

plt.hist(img_rescale.ravel(),50,[0,256])
plt.show()

equ = cv.equalizeHist(img)
plt.imshow(np.hstack((img,img_rescale,equ)), cmap='gray')
plt.xticks([]), plt.yticks([])

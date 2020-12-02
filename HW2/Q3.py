# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:40:07 2020

@author: toml
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# from scipy.signal import convolve2d as conv2
from numpy.fft import fft2, ifft2, ifftshift,fftshift
# from skimage import restoration
# from scipy import ndimage
# from numpy import sum,sqrt
# from numpy.random import standard_normal
# from skimage.util import random_noise 

woman = cv.imread('woman.jpg', cv.IMREAD_GRAYSCALE)
rect = cv.imread('rectangle.jpg', cv.IMREAD_GRAYSCALE)

plt.imshow(np.hstack((woman,rect)), cmap='gray')
plt.xticks([]), plt.yticks([])

wo_f = fftshift(fft2(woman))
re_f = fftshift(fft2(rect))

plt.imshow(20*np.log10(np.hstack( ( abs(wo_f), abs(re_f) ) ) ), cmap='gray')
plt.xticks([]), plt.yticks([])

plt.imshow(np.hstack( ( np.angle(wo_f), np.angle(re_f) ) ) , cmap='gray')
plt.xticks([]), plt.yticks([])

combined = abs(re_f) * np.exp(1j * np.angle(wo_f))

combined_img = abs(ifft2(combined));

# plt.imshow(np.hstack( ( woman, combined_img ) ) , cmap='gray')
plt.imshow(combined_img, cmap='gray')




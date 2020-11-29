# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:23:21 2020
This script is for the 2nd homework assigment
@author: toml
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE)

# kernel size: The greater the size, the more the motion. 
kernel_size = 10
  
# Create the kernals
kernel_v = np.zeros((kernel_size, kernel_size)) 
kernel_h = np.copy(kernel_v) 
  
# Filling of the the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
  
# Normalize kernals. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 
  
# convolution of the image with the kernal
vertical_mb = cv.filter2D(img, -1, kernel_v) 
horizonal_mb = cv.filter2D(img, -1, kernel_h) 

# Plot
plt.subplot(121),plt.imshow(img,cmap='gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(vertical_mb,cmap='gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
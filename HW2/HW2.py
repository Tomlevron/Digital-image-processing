# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:23:21 2020
This script is for the 2nd homework assigment
@author: toml
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2

img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE)

def create_kernal(size,horizontal=False):
    '''This function is creating a normalized kernal
    kernal is vertical by defult, enter False for horizntal'''
    kernel = np.zeros((size, size))
    
    if horizontal: #vertical
        kernel[:, int((size - 1)/2)] = np.ones(size)
    else: #horizontal
        kernel[int((size - 1)/2), :] = np.ones(size)  
    
    kernel /= size 
    return kernel
  
# Create the kernals
kernel_v = create_kernal(10)
kernel_h = create_kernal(10, horizontal=True) 
  
# convolution of the image with the kernal
vertical_mb = conv2(img, kernel_v, mode='same') 
horizontal_mb = conv2(img, kernel_h, mode='same') 

both = conv2(img, kernel_v, mode='same')
blur = cv.blur(img,(10,10))


# Plot
plt.figure
plt.imshow(img,cmap='gray'),plt.title('Orignal')
plt.xticks([]), plt.yticks([])
plt.show()


plt.figure
plt.subplot(121),plt.imshow(blur,cmap='gray'),plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(both,cmap='gray'),plt.title('conv2d')
plt.xticks([]), plt.yticks([])
plt.show()

plt.figure
plt.subplot(121),plt.imshow(horizontal_mb,cmap='gray'),plt.title('Horzintal')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(vertical_mb,cmap='gray'),plt.title('Vertical')
plt.xticks([]), plt.yticks([])
plt.show()

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
from numpy.fft import fft2, ifft2
from skimage import restoration

img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE)

def create_kernal(size,mode='vert',d=None):
    '''This function is creating a normalized kernal
    kernal is vertical by defult, enter False for horizntal'''
    if d == None:
        kernel = np.zeros((size, size))
        if mode == 'vert': #vertical
            kernel[:, int((size - 1)/2)] = np.ones(size)
        elif mode == 'horz': #horizontal
            kernel[int((size - 1)/2), :] = np.ones(size)  
        elif mode == 'both':
            kernel[:, int((size - 1)/2)] = np.ones(size)
            kernel[int((size - 1)/2), :] = np.ones(size)
    
    else:
        if mode == 'vert': #vertical
            kernel = np.ones((size, 1))
        elif mode == 'horz': #horizontal
            kernel = np.ones((1, size))
    kernel /= size 
    return kernel

def wiener_filter(img, kernel, noise):
    
    kernel /= np.sum(kernel)
    img_copy = np.copy(img)
    S_uu = fft2(img_copy, s = img.shape) # image spectrum
    S_nn = fft2(noise, s = img.shape) #noise spectrum
    gamma = S_nn / S_uu
    
    img_fft = fft2(img_copy)
    kernel_fft = fft2(kernel, s = img.shape)
    G = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + gamma)
    filterd_fft = img_fft * G
    filterd = np.abs(ifft2(filterd_fft))
    
    return filterd

# Create the kernals
kernel_v = create_kernal(10,d=1)
kernel_h = create_kernal(10, mode='horz') 
kernel_both = create_kernal(11,mode='both')

# convolution of the image with the kernal
vertical_mb = conv2(img, kernel_v, mode='same') 
horizontal_mb = conv2(img, kernel_h, mode='same') 
both = conv2(img, kernel_v, mode='same')
blur = cv.blur(img,(11,11))

wiener = wiener_filter(img, kernel_v, vertical_mb)
# plot
display = [img, blur, vertical_mb, wiener]
label = ['Original Image', 'Motion Blurred Image','vertical_mb', 
         'Wiener Filter applied']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(display[i], cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label[i])

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:28:39 2020

@author: toml
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from numpy.fft import fft2, ifft2, ifftshift
from skimage import restoration
from scipy import ndimage
from numpy import sum,sqrt
from numpy.random import standard_normal

img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]

def blur_image(img, kernel):
    '''
    Parameters
    ----------
    img : Array of uint8 (image of choice)
    kernel : Array of float/uint8 the same size as the image
    
    Returns
    -------
    im_blur : array of uint8 (the blurred input image)
    '''
    freq = fft2(img)
    freq_kernel = fft2(ifftshift(kernel))
    convolved1 = freq * freq_kernel
    im_blur = ifft2(convolved1).real
    im_blur = 255 * im_blur / np.max(im_blur)
    return im_blur

def create_kernal(size,mode='vert',d=None,im=img):
    '''This function is creating a normalized kernal
    kernal is vertical by defult, enter False for horizntal.
    if d is not None, the kernel is padded with zeros'''
    if d == None:
        kernel = np.zeros((size, size))
        if mode == 'vert': #vertical
            kernel = np.ones((size, 1))
            # kernel[:, int((size - 1)/2)] = np.ones(size)
        elif mode == 'horz': #horizontal
            kernel = np.ones((1, size))
            # kernel[int((size - 1)/2), :] = np.ones(size)  
        elif mode == 'both':
            kernel[:, int((size - 1)/2)] = np.ones(size)
            kernel[int((size - 1)/2), :] = np.ones(size)
    else:
        if mode == 'vert': #vertical
            kernel = np.zeros((size, size))
            kernel[:, int((size-1)/2)] = np.ones(size)
            kernel = kernel / size
            kernel = np.pad(kernel, (((im.shape[0]-size)//2,(im.shape[0]-size)//2),
                                     ((im.shape[1]-size)//2,(im.shape[1]-size)//2)),
                            padwithzeros)
        elif mode == 'horz': #horizontal
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            kernel = np.pad(kernel, (((im.shape[0]-size)//2,(im.shape[0]-size)//2),
                                     ((im.shape[1]-size)//2,(im.shape[1]-size)//2)),
                            padwithzeros)
    kernel /= size 
    return kernel

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def inverse_filter(img,im_blur,kernel):
    ''' Let f be the original image, h the blurring kernel, and g the blurred image. 
    The idea in inverse filtering is to recover the original image from the blurred image.
    '''
    epsilon = 10**-6
    freq_blur = fft2(im_blur)
    if im_blur.shape[0] != kernel.shape[0]:
        kernel = np.zeros((im_blur.shape[0],im_blur.shape[0])
        kernel[int(img.shape[0]/2 -5):int(img.shape[0]/2 +5), 343] = np.ones(10) /10
        
    freq_kernel = fft2( ifftshift(kernel) )
    freq_kernel = 1 / (epsilon + freq_kernel) # small numbers
 
    convolved = np.dot(freq_blur,freq_kernel) #freq_blur * freq_kernel
    restored = ifft2(convolved).real
    restored = 255 * restored / np.max(restored)
    return restored

# Create the kernals
kernel_full = create_kernal(10,d=1) *10
kernel_col = create_kernal(10) 

img_blur = blur_image(img, kernel_full)

# convolution of the image with the kernal
img_blur_conv = conv2(img, kernel_col, mode='same') 

restored_inverse = inverse_filter(img, img_blur, kernel_col)

# plt.imshow(img_blur,cmap='gray'), plt.xticks([]), plt.yticks([])
# plot
display = [img_blur, restored_inverse]
label = ['Blur function','inverse filter']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(display[i], cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label[i])

plt.show()






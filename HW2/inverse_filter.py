# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:10:54 2020

@author: tomse
"""
import numpy as np
import numpy.fft as fp
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d as conv2

def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

def inverse_filter(img,im_blur,kernel):
    
    epsilon = 10**-6
    freq_blur = fp.fft2(im_blur)
    freq_kernel = fp.fft2(fp.ifftshift(kernel))
    freq_kernel = 1 / (epsilon + freq_kernel) # small numbers
 
    convolved = freq_blur * freq_kernel
    im_restored = fp.ifft2(convolved).real
    im_restored = 255 * im_restored / np.max(im_restored)
    return im_restored
    
def blur_image(img, kernel):
    freq = fp.fft2(img)
    freq_kernel = fp.fft2(fp.ifftshift(kernel))
    convolved1 = freq * freq_kernel
    im_blur = fp.ifft2(convolved1).real
    im_blur = 255 * im_blur / np.max(im_blur)
    
    return im_blur

im = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE)
# create the motion blur kernel
size = 10
kernel = np.zeros((size, size))
kernel[:, int((size-1)/2)] = np.ones(size)
kernel = kernel / size

vertical_mb = conv2(im, kernel, mode='same') 

kernel = np.pad(kernel, (((im.shape[0]-size)//2,(im.shape[0]-size)//2),
                         ((im.shape[1]-size)//2,(im.shape[1]-size)//2)),
                padwithzeros)
 

im_blur = blur_image(im,kernel)
im_restored = inverse_filter(im,im_blur,kernel)
 
plt.figure(figsize=(18,12))
plt.subplot(221)
plt.imshow(im,cmap='gray')
plt.title('Original image', size=20)
plt.axis('off')
plt.subplot(222)
plt.imshow(im_blur,cmap='gray')
plt.title('Blurred image with motion blur kernel', size=20)
plt.axis('off')
plt.subplot(223)
plt.imshow(im_restored,cmap='gray')
plt.title('Restored image with inverse filter', size=20)
plt.axis('off')
plt.subplot(224)
plt.imshow(im_restored - im,cmap='gray')
plt.title('Diff restored & original image', size=20)
plt.axis('off')
plt.show()
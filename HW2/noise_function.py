# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 09:56:46 2020

@author: toml
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as conv2
from numpy.fft import fft2, ifft2, ifftshift,fftshift
from skimage import restoration
from scipy import ndimage
from numpy import sum,sqrt
from numpy.random import standard_normal
from skimage.util import random_noise 

# img = cv.imread('cameraman.tif', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]
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
    
    return im_blur.astype('uint8')


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
    freq_kernel = fft2( ifftshift(kernel) )
    freq_kernel = 1 / (epsilon + freq_kernel) # small numbers
 
    convolved = freq_blur * freq_kernel
    restored = ifft2(convolved).real
    restored = 255 * restored / np.max(restored)
    return restored.astype('uint8')

def add_noise(img, dB):
    # img = img.astype('uint8')
    img_var = ndimage.variance(img)
    img_mean = ndimage.mean(img)
    lin_SNR = 10.0 ** (dB/10.0)
    noise_var = img_var / lin_SNR 
    sigma_noise = (img_var / 10** ( dB / 10) ) **0.5
    print(noise_var)
    row,col = img.shape
    mean = 0.0
    sigma = noise_var ** 0.5
    print(sigma_noise)
    gauss = sigma * np.random.normal(img_mean,1,(row,col))  
    noisy = img + gauss
    noise_vari = img_var = ndimage.variance(img) / ndimage.variance(noisy)
    print('Noise vari is:')
    print(noise_vari)
    print('SNR ratio is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(gauss)))
    noisy = 255 * noisy / np.max(noisy)
    print('SNR ratio after normalization is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(gauss)))
    
    return noisy.astype('uint8')

size = 10
kernel = np.zeros((size, size))
kernel[:, int((size-1)/2)] = np.ones(size)
kernel = kernel / size 
kernel_v = np.copy(kernel)
kernel = np.pad(kernel, (((img.shape[0]-size)//2,(img.shape[0]-size)//2),
                         ((img.shape[1]-size)//2,(img.shape[1]-size)//2)),
                padwithzeros)
img_blur = blur_image(img, kernel)
a = add_noise(img_blur,20) 
plt.imshow(a,cmap='gray')
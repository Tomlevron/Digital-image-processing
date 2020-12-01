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
    return im_blur #.astype('uint8')


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
    return restored #.astype('uint8')

def add_noise(img, dB):
    img = img#.astype('uint8')
    img_var = ndimage.variance(img)
    lin_SNR = 10.0 ** (dB/10.0)
    # dB 10 * np.log10(img_var/noise_var)
    noise_var = img_var / lin_SNR #62.5 is needed
    sigma_noise = (img_var / 10** ( dB / 10) ) **0.5
    print(noise_var)
    row,col = img.shape
    mean = 0.0
    var = noise_var
    sigma = var**0.5
    print(sigma_noise)
    gauss = sigma * np.random.normal(mean,sigma,(row,col)).astype('uint8')
    # gauss = gauss.reshape(row,col)
    noisy = img + gauss
    print('SNR ratio is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(noisy)))
    noisy = 255 * noisy / np.max(noisy)
    print('SNR ratio is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(noisy)))
    return noisy.astype('uint8') 

size = 10
kernel = np.zeros((size, size))
kernel[:, int((size-1)/2)] = np.ones(size)
kernel = kernel / size 

kernel = np.pad(kernel, (((img.shape[0]-size)//2,(img.shape[0]-size)//2),
                         ((img.shape[1]-size)//2,(img.shape[1]-size)//2)),
                padwithzeros)

img_blur = blur_image(img, kernel)

restored_inverse = inverse_filter(img, img_blur, kernel)

plt.imshow(img_blur,cmap='gray'), plt.xticks([]), plt.yticks([])
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






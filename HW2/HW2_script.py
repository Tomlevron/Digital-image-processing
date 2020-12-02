# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:28:39 2020

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
    
    return im_blur#.astype('uint8')

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
    restored = abs(ifft2(convolved))#.real
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
    gauss = sigma * np.random.normal(0,1,(row,col))
    # gauss = 100 * np.random.normal(0,1,(row,col))  
    noisy = img + gauss
    noise_vari = img_var = ndimage.variance(img) / ndimage.variance(noisy)
    print('Noise vari is:')
    print(noise_vari)
    print('SNR ratio is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(gauss)))
    noisy = 255 * noisy / np.max(noisy)
    img = 255 * img / np.max(img)
    gauss = 255 * gauss / np.max(gauss)
    print('SNR ratio after normalization is:')
    print(10 * np.log10(ndimage.variance(img)/ndimage.variance(gauss)))
    
    return noisy#.astype('uint8')

def wiener_filter(img, kernel, noise):
    '''Perform winer filter on a given 
    image (img) with kernel (kernel) 
    and blurry image (noise) '''
    kernel /= np.sum(kernel) # h
    img_copy = np.copy(img).astype('uint8')
    S_uu = fft2(img_copy, s = img.shape) # image spectrum
    S_nn = fft2(noise, s = img.shape) #noise spectrum
    gamma = S_nn / S_uu
    
    img_fft = fft2(img_copy)
    kernel_fft = fft2(kernel, s = img.shape) # H
    G = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + gamma)
    filterd_fft = img_fft * G
    filterd = np.abs(ifft2(filterd_fft))
    
    return filterd.astype('uint8')

def wiener_filter_approx(img, kernel,k1):
    '''Perform approx winer filter on a given 
    image (img) with kernel (kernel) 
    and blurry image (noise) '''
    kernel /= np.sum(kernel)
    img_copy = np.copy(img)
    gamma = k1 #S_nn / S_uu
    
    img_fft = fft2(img_copy)
    kernel_fft = fft2(kernel, s = img.shape)
    G = np.conj(kernel_fft) / (np.abs(kernel_fft) ** 2 + gamma)
    filterd_fft = img_fft * G
    filterd = np.abs(ifft2(filterd_fft))
    # filterd = 255 * filterd / np.max(filterd)
    return filterd.astype('uint8')

def create_kernel(img, size):
    ''' Create two kernels for vertical motion blurring.
    kernel is padded with zeros, kernel_v is not.
    Parameters
    ----------
    img : input image
        input image for sizing of the kernel.
    size : size of the required kernel.

    Returns
    -------
    kernel : array of float 64 
        padded with zeros.
    kernel_v : array of float 64
        Not padded!

    '''
    # size = 10
    kernel = np.zeros((size, size))
    kernel[:, int((size-1)/2)] = np.ones(size)
    kernel = kernel / size 
    kernel_v = np.copy(kernel)
    kernel = np.pad(kernel, (((img.shape[0]-size)//2,(img.shape[0]-size)//2),
                             ((img.shape[1]-size)//2,(img.shape[1]-size)//2)),
                    padwithzeros)
    return kernel, kernel_v

# img = cv.imread('cameraman.tif', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]
img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]

kernel, kernel_v = create_kernel(img,10)

img_blur = blur_image(img, kernel)

restored_inverse = inverse_filter(img, img_blur, kernel)

img_blur_noisy = add_noise(img_blur,20)

restored_inverse_noisy = inverse_filter(img_blur, img_blur_noisy, kernel)

wiener = wiener_filter(img_blur_noisy, kernel_v, img_blur_noisy)

plt.imshow(img_blur,cmap='gray'), plt.xticks([]), plt.yticks([])
# plot
display = [img_blur, restored_inverse,restored_inverse_noisy,wiener]
label = ['Blur function','inverse filter','restored inverse noisy', 'wiener']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(display[i], cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label[i])

plt.show()

fig = plt.figure(figsize=(12, 10))
k = [0.1, 0.01, 0.001, 0.0001]
label = [str(i) for i in k]
for i in range(len(k)):
    wiener_approx = wiener_filter_approx(img_blur_noisy, kernel_v, k[i])
    fig.add_subplot(2, 2, i+1)
    plt.imshow(wiener_approx, cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label[i])

plt.show()



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
from numpy.fft import fft2, ifft2, ifftshift
from skimage import restoration


img = cv.imread('Gonz.jpg', cv.IMREAD_GRAYSCALE) # BGR and [Cols,Rows]

def add_noise(img, dB):
    
    lin_SNR = 10.0 ** (dB/10.0)
    
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
    return restored
   
#plt.imshow( (20*np.log10( 0.1 + np.fft.fftshift(freq))).astype(int), cmap='jet')


def create_kernal(size,mode='vert',d=None,im=img):
    '''This function is creating a normalized kernal
    kernal is vertical by defult, enter False for horizntal.
    if d is not None, the kernel is padded with zeros'''
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
            kernel = np.zeros((size, size))
            kernel[int((size-1)/2), :] = np.ones(size)
            kernel = kernel / size
            kernel = np.pad(kernel, (((im.shape[0]-size)//2,(im.shape[0]-size)//2+1),
                                     ((im.shape[1]-size)//2,(im.shape[1]-size)//2+1)),
                            padwithzeros)
        elif mode == 'horz': #horizontal
            kernel = np.zeros((size, size))
            kernel[:, int((size-1)/2)] = np.ones(size)
            kernel = kernel / size
            kernel = np.pad(kernel, (((im.shape[0]-size)//2,(im.shape[0]-size)//2+1),
                                     ((im.shape[1]-size)//2,(im.shape[1]-size)//2+1)),
                            padwithzeros)
    kernel /= size 
    return kernel

def wiener_filter(img, kernel, noise):
    '''Perform winer filter on a given 
    image (img) with kernel (kernel) 
    and blurry image (noise) '''
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

def blur_image(img, kernel):
    freq = fft2(img)
    freq_kernel = fft2(ifftshift(kernel))
    convolved1 = freq * freq_kernel
    im_blur = ifft2(convolved1).real
    im_blur = 255 * im_blur / np.max(im_blur)
    
    return im_blur
# Create the kernals
kernel_v = create_kernal(10)
kernel_h = create_kernal(10, mode='horz') 

# convolution of the image with the kernal
# vertical_mb = conv2(img, kernel_v, mode='same') 
# horizontal_mb = conv2(img, kernel_h, mode='same') 

size = 10
kernel = np.zeros((size, size))
kernel[:, int((size-1)/2)] = np.ones(size)
kernel = kernel / size 

kernel = np.pad(kernel, (((img.shape[0]-size)//2,(img.shape[0]-size)//2),
                         ((img.shape[1]-size)//2,(img.shape[1]-size)//2)),
                padwithzeros)

img_blur = blur_image(img,kernel)
inverse_restored = inverse_filter(img,img_blur,kernel)
wiener = wiener_filter(img, kernel_v, img_blur)
# plot
display = [img, img_blur,inverse_restored, wiener]
label = ['Original Image','Blurred','inverse restored', 
         'Wiener Filter applied']

fig = plt.figure(figsize=(12, 10))

for i in range(len(display)):
    fig.add_subplot(2, 2, i+1)
    plt.imshow(display[i], cmap = 'gray')
    plt.xticks([]), plt.yticks([])
    plt.title(label[i])

plt.show()

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


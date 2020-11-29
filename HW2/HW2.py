# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 10:23:21 2020
This script is for the 2nd homework assigment
@author: toml
"""
import numpy as np
import cv2

img = cv2.imread('Gonz.jpg', cv2.IMREAD_GRAYSCALE)

# kernel size: The greater the size, the more the motion. 
kernel_size = 10
  
# Create the vertical kernel. 
kernel_v = np.zeros((kernel_size, kernel_size)) 
  
# Create a copy of the same for creating the horizontal kernel. 
kernel_h = np.copy(kernel_v) 
  
# Fill the middle row with ones. 
kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size) 
kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size) 
  
# Normalize. 
kernel_v /= kernel_size 
kernel_h /= kernel_size 
  
# Apply the vertical kernel. 
vertical_mb = cv2.filter2D(img, -1, kernel_v) 
  
# Apply the horizontal kernel. 
horizonal_mb = cv2.filter2D(img, -1, kernel_h) 


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blurred',vertical_mb)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Blurred',horizonal_mb)
cv2.waitKey(0)
cv2.destroyAllWindows()

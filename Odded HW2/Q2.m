close all
clear all

I = imread('cameraman.tif');
I_dark = I./3; % darkened the image
figure(); imshow(I_dark); title('Darkened image in gray scale')
%% Q2 - 1
figure(); histogram(I_dark); title('Darkened image histogram')

%% Q2 - 2
contrast_stretch = imadjust(I_dark);
figure(); histogram(contrast_stretch); title('Stretched contrast image histogram') 
figure(); imshow(contrast_stretch); title('Stretched contrast image')

%% Q2 - 3
equalized_img = histeq(I_dark);
figure(); histogram(equalized_img); title('Equalized image histogram')
figure(); imshow(equalized_img); title('Equalize image')
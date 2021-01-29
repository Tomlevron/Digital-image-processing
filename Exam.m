clc;clear;close all;
img = rgb2gray(imread('A.png'));
% imshow(img)

A = [1 1 1];
B = [1 ; 1 ; 1];

conv_A = conv2(img,A,'same');
conv_B = conv2(img,B,'same');

figure;
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(A)
subplot(2,2,3)
imshow(uint8(conv_A))
title('Horz')
subplot(2,2,4)
imshow(uint8(conv_B))
title('Vert')

C = [1 -1];
D = [1;-1];

conv_C = conv2(img,C,'same');
conv_D = conv2(img,D,'same');

figure;
subplot(2,2,1)
imshow(img)
subplot(2,2,2)
imshow(C)
subplot(2,2,3)
imshow(uint8(conv_C))
title('Horz')
subplot(2,2,4)
imshow(uint8(conv_D))
title('Vert')
%%
img_5 = [ 4 4 8 8 8 8 8 8 8 8; 4 4 8 8 22 8 8 8 8 8; 4 4 8 8 22 8 8 8 8 8; ...
    4 4 4 20 20 20 7 7 7 7; 4 4 4 20 20 20 7 7 7 7; 4 4 4 20 20 20 7 7 7 7; ...
    5 5 5 17 5 18 7 7 7 7; 5 5 5 17 5 18 7 7 7 7; 5 5 5 17 5 18 7 7 7 7; ...
    5 5 5 5 5 5 7 7 7 7];


BW = imbinarize(img_5, 12);

SE = strel('square', 3);
eroded = imerode(BW, SE);

figure;
subplot(2,2,1)
imshow(uint8(img_5))
subplot(2,2,2)
imshow(BW)
subplot(2,2,3)
imshow(eroded)
subplot(2,2,4)
imshow(BW - eroded)



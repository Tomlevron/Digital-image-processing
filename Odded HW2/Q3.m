close all
clear all

I1 =  (imread('trees.tif'));
I2 =  (imread('tire.tif'));

I2 = imresize(I2, size(I1));

figure(); imshow(I1); title('Trees')
figure(); imshow(I2); title('Tire')

I1_F = fft2(I1);
I2_F = fft2(I2);

I_new_F = abs(I1_F).*exp(1i*angle(I2_F)); %construct the new image

I_new = ifft2(I_new_F);
figure(); imshow(uint8(I_new)); title('Amplitude: Trees, phase: Tire')

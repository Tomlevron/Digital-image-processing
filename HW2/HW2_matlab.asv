%% Lab HW2 Tom Lev-ron
clc;clear;
img = (double (imread('Gonz.jpg')) );
kernel = ones(10,1) ./10;

img_blur = conv2(img,kernel,'same');
figure;
imshow(uint8(img_blur),[])

% Inverse filter

freq_img = fft2(img_blur);
freq_kernel = fft2(kernel,size(img_blur,1),size(img_blur,2));
figure;
imshow(ifft2(freq_img .* freq_kernel),[])

inverse_filter = freq_img ./ freq_kernel;
inverse_time = real(ifft2(inverse_filter));
figure;
imshow(uint8(inverse_time))
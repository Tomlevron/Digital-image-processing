close all; clear all;
% load image and plot image
img = double(imread('Gonz.jpg'));
figure() ; imshow(uint8(img),[]);  title('Original image in gray scale')

%% Q1 - 1
filter = ones(1,10)./10; % calculate the filter
filter_F = (fft2(filter, size(img, 1), size(img, 2))); % fourier transform the filter
img_F = fft2(img); % fourier transform the image
bluredI_F = img_F.*filter_F; % convolution in the spatial frequency domain => perform a multipicationi
bluredI = ifft2(bluredI_F); % blured image in the image domain
figure() ;imshow(bluredI, []) ; title('Blurred image')

%% Q1 - 2
bluredI_backto_F = fft2(bluredI); % fourier transform the blured image
filter_F( find(filter_F==0) ) = eps; % create a constrained inverse filter
invFilter_F = 1./filter_F;
res_F = bluredI_backto_F.*invFilter_F; %restored image in the frequency domain
res = (ifft2(res_F));   % restored blurred image in the image domain
figure() ;  imshow(res,[]); title('Restored Blurred image with an inverse filter')

%%  Q1 - 3
% calculate the added noise
SNR = 20;
img_var = var(img(:));
noise_var = img_var/( 10^(0.1*SNR));
added_noise =  sqrt(noise_var)*randn(size(img));
bluredI_noise = bluredI + added_noise; % add the noise to the blurred image
figure(); imshow(bluredI_noise, []); title('Blurred image with added noise')
% restore the blured and noisy image using the inverse filter
bluredI_noise_F = fft2(bluredI_noise);  % fourier transform the blurred and noisy image
bluredI_noise_F_restored = bluredI_noise_F.*invFilter_F;  % apply the inverse filter on thef blurred and noisy image in frequency domain
bluredI_noise_res = ifft2(bluredI_noise_F_restored); %restored the blurred and noisy image in the image domain
figure(); imshow(bluredI_noise, []); title('Restored Blurred and noisy image with an inverse filter')

%%  Q1 - 4.a

% calculate wiener filter
Suu = imgSpectrum(img);
conj_H = conj(filter_F);
sizeH_sqr = abs(filter_F).^2;
Snini = imgSpectrum(added_noise);

winner_num = conj_H.*Suu;
winner_den = sizeH_sqr.* Suu + Snini;
Winner = winner_num./winner_den;

res_Winner_F = bluredI_noise_F.*Winner; % apply wiener filter in the fourier domain
res_Winner = ifft2(res_Winner_F);
figure(); imshow(res_Winner, []); title('Restored Blurred and noisy image with Wiener filter')


%%  Q1 - 4.b
  % calculate approximated wiener filter
for gamma = [0, 0.1, 1, 10]
    G = conj_H ./ (sizeH_sqr + gamma);
    res_Winner_F = bluredI_noise_F.*G;
    res_Winner = ifft2(res_Winner_F);
    
    figure(); imshow(res_Winner, []); set(gcf, 'Position',  [100, 100, 400, 400])
    title({'Restored Blurred and noisy image';['with an aproximated Wiener filter, gamma=', num2str(gamma)]})
end
function [ spectrum ] = imgSpectrum( img )

%calculate 2D fft
fft_img = fft2(img);

%calc fft magnitude
spectrum = (abs(fft_img)).^2;

end


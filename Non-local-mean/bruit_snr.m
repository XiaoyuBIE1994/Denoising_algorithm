%%% This function is used to add a gaussian noise with a given SNR to a Gray image
%%% 
%%% Input:
%%% im : the given image as an imput
%%% snr : value of SNR (dB)
%%% Output:
%%% y : noisy image with a SNR
%%% VAR_bruit: variance of the noise    
%%% 

function [y,VAR_bruit]=bruit_snr(im,snr)

% conversion of iamge with 1D vector
tab = size(im);
tab_im = reshape(im,1,tab(1)*tab(2));

% Variance of the iamge
VAR_im = var(tab_im);

% Calculate the variance of the gaussian noise 

VAR_bruit = VAR_im * 10^(-snr/10);

% Add noise to the input image

y = im + sqrt(VAR_bruit)*randn(tab(1),tab(2));




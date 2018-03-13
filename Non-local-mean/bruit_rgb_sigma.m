%%% This function is used to add a gaussian noise to a RGB image
%%% Input:
%%% im : image
%%% sigma :value of noise
%%% Output:
%%% y : noisy image, variance of noise is sigma


function y=bruit_rgb_sigma(im,sigma)

% conversion of tableau with 3D vecter
tab=size(im);
bruit1=sigma*randn(tab(1),tab(2));
bruit2=sigma*randn(tab(1),tab(2));
bruit3=sigma*randn(tab(1),tab(2));

% add noise to the imput image
y(:,:,1) = im(:,:,1) + bruit1;
y(:,:,2) = im(:,:,2) + bruit2;
y(:,:,3) = im(:,:,3) + bruit3;
end
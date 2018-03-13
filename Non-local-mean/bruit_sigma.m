%%% This function is used to add a gaussian noise to a Gray image
%%% Input:
%%% im : image
%%% sigma :value of noise
%%% Output:
%%% y : image bruit¨¦ avec un rapport signal sur bruit


function y=bruit_sigma(im,sigma)

% conversion of tableau with 3D vecter
tab=size(im);
bruit=sigma*randn(tab(1),tab(2));

% add noise to the imput image
y = im + bruit;
end


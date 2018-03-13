%%% This is a function to calculate the PSNR for a RBG image
%%% input: I: noisy image
%%%        I_restore: restored image
%%% output:PSNR 
function PSNR = PSNR_rgb_out( I,I_restore )
[m,n,p]=size(I);
G=1/(3*m*n);
MSE=G*(sum(sum((I_restore(:,:,1)-I(:,:,1)).^2))+sum(sum((I_restore(:,:,2)-I(:,:,2)).^2))+sum(sum((I_restore(:,:,3)-I(:,:,3)).^2)));
PSNR=10*log10(255^2/MSE);
end

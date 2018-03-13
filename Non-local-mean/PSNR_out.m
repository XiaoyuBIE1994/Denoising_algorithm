%%% This is a function to calculate the PSNR for a gray image
%%% input: I: noisy image
%%%        I_restore: restored image
%%% output:PSNR 
function PSNR = PSNR_out( I,I_restore )
[m,n]=size(I);
G=1/(m*n);
MSE=G*sum(sum((I_restore-I).^2));
PSNR=10*log10(255^2/MSE);
end


function y=MSE_image(I,M)
tab=size(I);
y=sum(sum(((I-M).^2)))/(tab(1)*tab(2));
end
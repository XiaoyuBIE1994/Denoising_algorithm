function y=var_image(I)
tab=size(I);
tab_I=reshape(I,1,tab(1)*tab(2));
y=var(tab_I);
end
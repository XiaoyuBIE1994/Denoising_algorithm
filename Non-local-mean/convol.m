function y=convol(h,im)

%%% usage y=convol(filtre,image)
%%%
%%% Ce programme realise la convolution d'un filtre
%%% et d'une image d'entree, qui sont des matrices de
%%% tailles eventuellement differentes.


%%% Determination du maximum des dimemsions des deux images 

  N=max(size(im,1),size(h,1));
  M=max(size(im,2),size(h,2));

%%% Determination du centre de gravite du filtre
%%% PAS BESOIN D'ANLYSER CETTE PARTIE
% En x -> x_grav

  i=1:size(h,2);
  j=ones(size(h,2),1);
  k=kron(j,i);
  x_grav=round(sum(sum(abs(h).*k))/sum(sum(abs(h))));

% En y -> y_grav

  i=1:size(h,1);
  j=ones(size(h,1),1);
  k=kron(j,i);
  y_grav=round(sum(sum(abs(h).*(k')))/sum(sum(abs(h)))); 
  
  
%%% Calcul des FFT de l'image et du filtres. Notez que les deux 
%%% images d'entree sont "plongees" dans un image de dimensions
%%% NxM. Les pixels "en trop" sont mis a zero (zero-padding)

  a=fft2(h,N,M);
  b=fft2(im,N,M);

%%% Calcul de la FFT inverse du produit des FFT -> produit de convolution

  z=real(ifft2(a .* b));
  
%%% Decalage du produit de convolution pour le centrer 
%%% sur l'image de depart

  y=shift(shift(z,1-y_grav)',1-x_grav)'; 

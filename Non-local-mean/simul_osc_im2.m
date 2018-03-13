%%% This profile is use to study the performance of NL-mean in complex OSC
%%% images
%% Initialization
%%%
%%%
%%%
%%%
%%% delta_P=0.2
%%% sig=1
%%% 
%%% SIMULATION D'IMAGES OSC
clc
clear all   
close all
%%% import the image
I=(double(rgb2gray(imread('sunset.jpg')))+1)*0.9;
tab=size(I);
figure(1)
imagesc(I); colormap gray
title('Original image')
%%
% Definition of the size of target0
target_size=12; % 25*25 
target_x1=300;
target_y1=200;
target_x2=600;
target_y2=250;
target_x3=800;
target_y3=500;
% Varicance of noise
sig =10;

% OSC
P=0.5; % OSC mean
delta_P=0.2; % Difference d'OSC
P_a = P + delta_P/2 ;  % OSC target
P_b = P - delta_P/2 ;  % OSC background

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
      
% Image target
target=P_b*ones(tab(1),tab(2));
target(target_y1-target_size:target_y1+target_size,target_x1-target_size:target_x1+target_size)=P_a;
target(target_y2-target_size:target_y2+target_size,target_x2-target_size:target_x2+target_size)=P_a;
target(target_y3-target_size:target_y3+target_size,target_x3-target_size:target_x3+target_size)=P_a;
%%% Generation des images X et Y
X_original= I .* (1+target)/2;
Y_original= I .* (1-target)/2;
X = I .* (1+target)/2+ sig*randn(tab(1),tab(2));
Y = I .* (1-target)/2+ sig*randn(tab(1),tab(2));
I_im_original=X_original+Y_original;
P_im_original=(X_original-Y_original)./(X_original+Y_original);
I_im = X+Y;
P_im = (X-Y)./(X+Y);
P_im(find(P_im>1))=1; % delete the overflow
P_im(find(P_im<0))=0;
figure(2); imagesc(I_im_original); colormap gray;
figure(3); imagesc(P_im_original); colormap gray;
figure(4); imagesc(I_im); colormap gray;
figure(5); imagesc(P_im); colormap gray;
Intensity_means=sum(sum(I_im_original))/(tab(1)*tab(2));
SNR=10*log10(Intensity_means/sig)
%% Filtering
X_new=NLmeans_grey(X,sig);
Y_new=NLmeans_grey(Y,sig);
% X_new=NLmeans_grey(X_new,sig);
% Y_new=NLmeans_grey(Y_new,sig);
I_im_new=X_new+Y_new;
P_im_new= (X_new-Y_new)./(X_new+Y_new);
P_im_new(find(P_im_new>1))=1; % delete the overflow
P_im_new(find(P_im_new<0))=0;
figure(6); imagesc(I_im_new); colormap gray;
figure(7); imagesc(P_im_new); colormap gray;
%%
figure()
imagesc(I_im_original); colormap gray; axis off;
figure()
imagesc(I_im); colormap gray; axis off;
figure()
imagesc(I_im_new); colormap gray; axis off;
figure()
imagesc(P_im_original); colormap gray; axis off;
figure()
imagesc(P_im); colormap gray; axis off;
figure()
imagesc(P_im_new); colormap gray; axis off;
MSE1=MSE_image(P_im_new,P_im_original);
%% target only
Pad_target=I_im(target_y-target_size:target_y+target_size,target_x-target_size:target_x+target_size);
Intensity_target=sum(sum(Pad_target))/((2*target_size+1)^2)
SNR=Intensity_target/(sqrt(2)*sig)
target_original=X_original(target_y-target_size:target_y+target_size,target_x-target_size:target_x+target_size);
target_noisy=X(target_y-target_size:target_y+target_size,target_x-target_size:target_x+target_size);
target_new=X_new(target_y-target_size:target_y+target_size,target_x-target_size:target_x+target_size);
RMSE_target_noisy=sqrt(MSE_image(target_noisy,target_original));
RMSE_target_new=sqrt(MSE_image(target_new,target_original));
r=RMSE_target_noisy/RMSE_target_new
%% target + background
target_x=target_x3;
target_y=target_y3;

Pad_P_new=P_im_new(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
Pad_P_noisy=P_im(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
Pad_P_original=P_im_original(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
Pad_target=I_im(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
Intensity_target=sum(sum(Pad_target))/((4*target_size+1)^2)
target_original=X_original(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
target_noisy=X(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);
target_new=X_new(target_y-2*target_size:target_y+2*target_size,target_x-2*target_size:target_x+2*target_size);

% RMSE for complex image, noisy and new, X and polarization
RMSE_target_noisy=sqrt(MSE_image(target_noisy,target_original));
RMSE_target_new=sqrt(MSE_image(target_new,target_original));
r=RMSE_target_noisy/RMSE_target_new
RMSE_P_noisy=sqrt(MSE_image(Pad_P_noisy,Pad_P_original));
RMSE_P_new=sqrt(MSE_image(Pad_P_new,Pad_P_original));
r_P=RMSE_P_noisy/RMSE_P_new

tab_Pad=size(Pad_target);
I_simple=Intensity_target*ones(tab_Pad(1),tab_Pad(2));
target_simple=P_b*ones(tab_Pad(1),tab_Pad(2));
target_simple(target_size+1:3*target_size+1,target_size+1:3*target_size+1)=P_a;
X_simple_original=I_simple.*(1+target_simple)/2;
Y_simple_original=I_simple.*(1-target_simple)/2;
X_simple_noisy=I_simple.*(1+target_simple)/2+sig*randn(tab_Pad(1),tab_Pad(2));
Y_simple_noisy=I_simple.*(1-target_simple)/2+sig*randn(tab_Pad(1),tab_Pad(2));
X_simple_new=NLmeans_grey(X_simple_noisy,sig);
Y_simple_new=NLmeans_grey(Y_simple_noisy,sig);
P_simple_original=(X_simple_original-Y_simple_original)./(X_simple_original+Y_simple_original);
P_simple_noisy=(X_simple_noisy-Y_simple_noisy)./(X_simple_noisy+Y_simple_noisy);
P_simple_noisy(find(P_simple_noisy>1))=1; % delete the overflow
P_simple_noisy(find(P_simple_noisy<0))=0;
P_simple_new=(X_simple_new-Y_simple_new)./(X_simple_new+Y_simple_new);
P_simple_new(find(P_simple_noisy>1))=1; % delete the overflow
P_simple_new(find(P_simple_noisy<0))=0;

% RMSE for simple image, noisy and new, X and polarization
RMSE_target_noisy_simple=sqrt(MSE_image(X_simple_noisy,X_simple_original));
RMSE_target_new_simple=sqrt(MSE_image(X_simple_new,X_simple_original));
r_simple=RMSE_target_noisy_simple/RMSE_target_new_simple
RMSE_P_noisy_simple=sqrt(MSE_image(P_simple_noisy,P_simple_original));
RMSE_P_new_simple=sqrt(MSE_image(P_simple_new,P_simple_original));
r_P_simple=RMSE_P_noisy_simple/RMSE_P_new_simple
%%
figure()
subplot(221)
imagesc(Pad_P_noisy); colormap gray; axis off;title('(a) Noisy partial image ')
subplot(222)
imagesc(P_simple_noisy); colormap gray; axis off;title('(b) Noisy partial image(simple) ')
subplot(223)
imagesc(Pad_P_new); colormap gray; axis off;title('(c) Restored partial image ')
subplot(224)
imagesc(P_simple_new); colormap gray; axis off;title('(d) Restored partial image(simple) ')
%%
RMSE_noisy=sqrt(MSE_image(X,X_original))
RMSE_new=sqrt(MSE_image(X_new,X_original))
r_total=RMSE_noisy/RMSE_new
%% pre-filtering estimation
%% average 2
X_new=NLmeans_grey_average2(X,sig);
Y_new=NLmeans_grey_average2(Y,sig);
P_im_new= (X_new-Y_new)./(X_new+Y_new);
P_im_new(find(P_im_new>1))=1; % delete the overflow
P_im_new(find(P_im_new<0))=0;
figure(10)
imagesc(P_im_new); colormap gray; axis off;
%% average 3
X_new=NLmeans_grey_average3(X,sig);
Y_new=NLmeans_grey_average3(Y,sig);
P_im_new= (X_new-Y_new)./(X_new+Y_new);
P_im_new(find(P_im_new>1))=1; % delete the overflow
P_im_new(find(P_im_new<0))=0;
figure(11)
imagesc(P_im_new); colormap gray; axis off;
%% gaussian 
X_new=NLmeans_grey_gaussian(X,sig);
Y_new=NLmeans_grey_gaussian(Y,sig);
P_im_new= (X_new-Y_new)./(X_new+Y_new);
P_im_new(find(P_im_new>1))=1; % delete the overflow
P_im_new(find(P_im_new<0))=0;
figure(12)
imagesc(P_im_new); colormap gray; axis off;
%% NL-mean
X_NL=NLmeans_grey(X,sig);
X_new=NLmeans_grey_doubleNL(X,sig,X_NL);
Y_NL=NLmeans_grey(Y,sig);
Y_new=NLmeans_grey_doubleNL(Y,sig,Y_NL);
P_im_new= (X_new-Y_new)./(X_new+Y_new);
P_im_new(find(P_im_new>1))=1; % delete the overflow
P_im_new(find(P_im_new<0))=0;
figure(13)
imagesc(P_im_new); colormap gray; axis off;
%%% his profile is used to study the performance of NL-mean denoising in
%%% RBG images
%%% In order to economize the computing time, we just choose a small piece
%%% in a large RBG image to evaluate
%%%
%% initialization_grey
close all
clear all
clc
sigma=10;
I=double(imread('sf2.gif')); % read th image
I_bruit=bruit_sigma(I,sigma);% add the noise with 10dB
% calculate its computation duration and SNR
tic
I_restore=NLmeans_grey(I_bruit,sigma);
t1=toc
PSNR=PSNR_out(I,I_restore)
figure(1)
subplot(221)
imshow(I/255)
subplot(222)
imshow(I_bruit/255)
subplot(223)
imshow(I_restore/255)

%% RGB reserve the figure 
close all
clear all
clc
figure(1)
%alley
I=imread('alley.png');
I=I(150:250,300:400,:);
subplot(241)
imshow(I)
imwrite(I,'alley_reduce.png');
%computer
I=imread('computer.png');
I=I(150:250,300:400,:);
subplot(242)
imshow(I)
imwrite(I,'computer_reduce.png');
%dice
I=imread('dice.png');
I=I(150:250,300:400,:);
subplot(243)
imshow(I)
imwrite(I,'dice_reduce.png');
%flowers
I=imread('flowers.png');
I=I(150:250,300:400,:);
subplot(244)
imshow(I)
imwrite(I,'flowers_reduce.png');
%girl
I=imread('girl.png');
I=I(150:250,300:400,:);
subplot(245)
imshow(I)
imwrite(I,'girl_reduce.png');
%traffic
I=imread('traffic.png');
I=I(150:250,300:400,:);
subplot(246)
imshow(I)
imwrite(I,'traffic_reduce.png');
%trees
I=imread('trees.png');
I=I(150:250,300:400,:);
subplot(247)
imshow(I)
imwrite(I,'trees_reduce.png');
%Valledemoussa
I=imread('Valldemoussa.png');
I=I(150:250,300:400,:);
subplot(248)
imshow(I)
imwrite(I,'Valldemoussa_reduce.png');
%% initialization_colour
close all
clear all
clc
l=[2,5,10,20,30,40];
for i=1:6
    sigma=l(i);
I=double(imread('Valldemoussa_reduce.png')); % read th image
I_bruit=bruit_rgb_sigma(I,sigma);% add the noise with 10dB
% calculate its computation duration and SNR
tic
I_restore=NLmeans_rgb(I_bruit,sigma);
t(i)=toc;
PSNR=PSNR_rgb_out(I,I_restore);
P(i)=PSNR;
end
P
t
%%
figure(1)
subplot(131)
imshow(I/255)
subplot(132)
imshow(I_bruit/255)
subplot(133)
imshow(I_restore/255)   
%% comparision
figure()
sigma=[2 5 10 20 30 40];
mat=[42.11 34.34 29.04 25.85 22.74 22.14];
ipol=[41.79 33.58 30.12 26.36 23.78 22.61];
plot(sigma,mat,'r',sigma,ipol,'b')
xlabel('\sigma')
ylabel('PSNR')
legend('Matlab','IPOL')
%% gausien kernel
close all;
clear all;
clc
l=[2,5,10,20,30,40];
for i=1:6
    sigma=l(i);
I=double(imread('girl_reduce.png')); % read th image
I_bruit=bruit_rgb_sigma(I,sigma);% add the noise with 10dB
% calculate its computation duration and SNR
if (sigma<=25)
    ds=1;
    Ds=10;
    h=0.55*sigma;
else if (sigma<=55)
        ds=2;
        Ds=10;
        h=0.4*sigma;
    else if (sigma<=100)
        ds=3;
        Ds=17;
        h=0.35*sigma;
        end
    end
end
tic
I_restore_1=NLmeans_rgb(I_bruit,ds,Ds,h,sigma);
t_1(i)=toc;
P_1(i)=PSNR_rgb_out(I,I_restore_1);
tic
I_restore_2=NLmeans_rgb_gaussian(I_bruit,ds,Ds,h,sigma);
t_2(i)=toc;
P_2(i)=PSNR_rgb_out(I,I_restore_2);
end
P_1
P_2
t_1
t_2
figure()
subplot(221)
imshow(I/255)
subplot(222)
imshow(I_bruit/255)
subplot(223)
imshow(I_restore_1/255)
subplot(224)
imshow(I_restore_2/255)
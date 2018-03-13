%%% This profile is used to study the influence of parameters in NL-means
%%% denoising algrithm.
%%% The iamge we used is a gray image in Internet
%%%
%% initialization
close all;
clear all;
clc
I=double(imread('sf2.gif')); % read th image
[I_bruit,var]=bruit_snr(I,10);% add the noise with 10dB
sig=sqrt(var);
% calculate its computation duration and SNR
%% Application of the algorithm
I_nlm=NLmeans(I_bruit,2,5,10,sig);
SNR1=SNR_out(I_bruit,I);
SNR2=SNR_out(I_nlm,I);
figure(1)
subplot(131)
imagesc(I)
subplot(132)
imagesc(I_bruit)
subplot(133)
imagesc(I_nlm)
colormap('gray')
%% influence of the research window
Ds=[3,4,5,6,7,8,9];
for i=1:7
    tic
    I_nlm=NLmeans(I_bruit,2,Ds(i),2,sig);
    time_1(i)=toc;
    SNR_1(i)=SNR_out(I_nlm,I);
    % imshow([I,I_nlm],[]);
end
figure(2)
subplot(121)
plot(2*Ds+1,time_1)
xlabel('size of research window')
ylabel('computation time(s)')
subplot(122)
plot(2*Ds+1,SNR_1)
xlabel('size of research window')
ylabel('SNR(dB)')
%% influence of the patch window
ds=[1,2,3,4];
for i=1:4
    tic
    I_nlm=NLmeans(I_bruit,ds(i),5,2,sig);
    time_2(i)=toc;
    SNR_2(i)=SNR_out(I_nlm,I);
    % imshow([I,I_nlm],[]);
end
figure(3)
subplot(121)
plot(2*ds+1,time_2)
xlabel('size of patch window')
ylabel('computation time(s)')
subplot(122)
plot(2*ds+1,SNR_2)
xlabel('size of patch window')
ylabel('SNR(dB)')
%% influence of the h
h=[2,3,4,5,6,7,8,9,10];
for i=1:9
    tic
    I_nlm=NLmeans(I_bruit,2,5,h(i),sig);
    time_3(i)=toc;
    SNR_3(i)=SNR_out(I_nlm,I);
    % imshow([I,I_nlm],[]);
end
figure(4)
subplot(121)
plot(h,time_3)
xlabel('coefficient h')
ylabel('computation time(s)')
subplot(122)
plot(h,SNR_3)
xlabel('coefficient h')
ylabel('SNR(dB)')
%% influence of the noise
noise=[10,15,20,25,30];
for i=1:5
    I_bruit=bruit_snr(I,noise(i));
    tic
    I_nlm=NLmeans(I_bruit,2,5,10,noise(i));
    time_4(i)=toc;
    SNR_4(i)=SNR_out(I_nlm,I);
    % imshow([I,I_nlm],[]);
end
figure(4)
subplot(121)
plot(noise,time_4)
xlabel('Bruit(dB)')
ylabel('computation time(s)')
subplot(122)
plot(noise,SNR_4)
xlabel('Bruit(dB)')
ylabel('SNR(dB)')
% input the image 

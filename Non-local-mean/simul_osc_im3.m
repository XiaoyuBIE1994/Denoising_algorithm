%%% This profile is use to study the performance of NL-mean in practical OSC
%%% images
%%% This program doesn't support the 
%% Initialization
close all
clear all
clc
load('I_par.mat')
load('I_per.mat')
I_par=I_par/100;
I_per=I_per/100;
I_noisy=(I_par-I_per)./(I_par+I_per);
figure(1)
imagesc(I_noisy); colormap gray; axis off
sig_assume=20;
%% noise research
% image par
figure(2)
imagesc(I_par); colormap gray;
I1=I_par(10:40,10:40);
M1=mean2(I1)
V1=sqrt(var_image(I1)) 
I2=I_par(70:100,100:130);
M2=mean2(I2)
V2=sqrt(var_image(I2))
I3=I_par(10:40,280:310);
M3=mean2(I3)
V3=sqrt(var_image(I3))
I4=I_par(210:240,10:40);
M4=mean2(I4)
V4=sqrt(var_image(I4))
figure(3)
subplot(221)
imagesc(I1); colormap gray;
subplot(222)
imagesc(I2); colormap gray;
subplot(223)
imagesc(I3); colormap gray;
subplot(224)
imagesc(I4); colormap gray;
figure(4)
plot([M4 M3 M1 M2],[V4 V3 V1 V2]); grid on
xlabel('Mean value')
ylabel('Variance')
% iamge per
figure(5)
imagesc(I_per); colormap gray;
I1=I_per(10:40,10:40);
M1=mean2(I1)
V1=sqrt(var_image(I1))
I2=I_per(70:100,100:130);
M2=mean2(I2)
V2=sqrt(var_image(I2))
I3=I_per(10:40,280:310);
M3=mean2(I3)
V3=sqrt(var_image(I3))
I4=I_per(210:240,10:40);
M4=mean2(I4)
V4=sqrt(var_image(I4))
figure(6)
subplot(221)
imagesc(I1); colormap gray;
subplot(222)
imagesc(I2); colormap gray;
subplot(223)
imagesc(I3); colormap gray;
subplot(224)
imagesc(I4); colormap gray;
figure(7)
plot([M4 M3 M1 M2],[V4 V3 V1 V2]); grid on
xlabel('Mean value')
ylabel('Variance')
%% NL
I_par_res=NLmeans_grey(I_par,10);
I_per_res=NLmeans_grey(I_per,10);
OSC_noisy=(I_par_res-I_per_res)./(I_par_res+I_per_res);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(2)
imagesc(OSC_noisy); colormap gray; axis off
%% NL+average2
I_par_res=NLmeans_grey_average2(I_par,10);
I_per_res=NLmeans_grey_average2(I_per,10);
OSC_noisy=(I_par_res-I_per_res)./(I_par_res+I_per_res);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(3)
imagesc(OSC_noisy); colormap gray; axis off
%% NL+average3
I_par_res=NLmeans_grey_average3(I_par,10);
I_per_res=NLmeans_grey_average3(I_per,10);
OSC_noisy=(I_par_res-I_per_res)./(I_par_res+I_per_res);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(4)
imagesc(OSC_noisy); colormap gray; axis off
%% NL+gaussian
I_par_res=NLmeans_grey_gaussian(I_par,10);
I_per_res=NLmeans_grey_gaussian(I_per,10);
OSC_noisy=(I_par_res-I_per_res)./(I_par_res+I_per_res);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(5)
imagesc(OSC_noisy); colormap gray; axis off
%% doubleNL
I_par_new=NLmeans_grey(I_par,10);
I_per_new=NLmeans_grey(I_per,10);
I_par_res=NLmeans_grey_doubleNL(I_par,10,I_par_new);
I_per_res=NLmeans_grey_doubleNL(I_per,10,I_per_new);
OSC_noisy=(I_par_res-I_per_res)./(I_par_res+I_per_res);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(6)
imagesc(OSC_noisy); colormap gray; axis off
%% Traditional filter
H=fspecial('gaussian');
I_par_new=imfilter(I_par,H);
I_per_new=imfilter(I_per,H);
OSC_noisy=(I_par_new-I_per_new)./(I_par_new+I_per_new);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure()
imagesc(OSC_noisy); colormap gray; axis off
%%
figure()
imagesc(I_noisy); colormap gray;
figure()
imagesc(OSC_noisy); colormap gray;
%%
figure()
imagesc(I_par); colormap gray;
figure()
imagesc(I_par_res); colormap gray;
%%
I_noisy1=I_noisy(300:400,300:400);
OSC_noisy=OSC_noisy(300:400,300:400);
figure()
subplot(121)
imagesc(I_noisy1); colormap gray; 
subplot(122)
imagesc(OSC_noisy); colormap gray; 

I_noisy2=I_noisy(200:250,300:350);
I_res2=OSC_noisy(200:250,300:350);
figure()
subplot(121)
imagesc(I_noisy2); colormap gray; 
subplot(122)
imagesc(I_res2); colormap gray; 
%% plant research (blur effect)
close all
clear all
clc
load('I_par.mat')
load('I_per.mat')
I_par=I_par/100;
I_per=I_per/100;
I_noisy=(I_par-I_per)./(I_par+I_per);
figure(1)
imagesc(I_noisy); colormap gray; axis off


sig_assume=[1 2 3 5 10 20 30];
%%% choose different sig to change the parameters 
% sig=1
sig_assume=1;
I_par_res1=NLmeans_grey(I_par,sig_assume);
I_per_res1=NLmeans_grey(I_per,sig_assume);
OSC_noisy=(I_par_res1-I_per_res1)./(I_par_res1+I_per_res1);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(2)
subplot(131)
imagesc(I_par_res1); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res1); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(OSC_noisy); colormap gray; axis off
title('I_{res}')
% sig=2
sig_assume=2;
I_par_res2=NLmeans_grey(I_par,sig_assume);
I_per_res2=NLmeans_grey(I_per,sig_assume);
I_res2=(I_par_res2-I_per_res2)./(I_par_res2+I_per_res2);
I_res2(find(I_res2>1))=1; % delete the overflow
I_res2(find(I_res2<0))=0;
figure(3)
subplot(131)
imagesc(I_par_res2); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res2); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res2); colormap gray; axis off
title('I_{res}')
% sig=3
sig_assume=3;
I_par_res3=NLmeans_grey(I_par,sig_assume);
I_per_res3=NLmeans_grey(I_per,sig_assume);
I_res3=(I_par_res3-I_per_res3)./(I_par_res3+I_per_res3);
I_res3(find(I_res3>1))=1; % delete the overflow
I_res3(find(I_res3<0))=0;
figure(4)
subplot(131)
imagesc(I_par_res3); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res3); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res3); colormap gray; axis off
title('I_{res}')
% sig=5
sig_assume=5;
I_par_res5=NLmeans_grey(I_par,sig_assume);
I_per_res5=NLmeans_grey(I_per,sig_assume);
I_res5=(I_par_res5-I_per_res5)./(I_par_res5+I_per_res5);
I_res5(find(I_res5>1))=1; % delete the overflow
I_res5(find(I_res5<0))=0;
figure(5)
subplot(131)
imagesc(I_par_res5); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res5); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res5); colormap gray; axis off
title('I_{res}')
% sig=10
sig_assume=10;
I_par_res10=NLmeans_grey(I_par,sig_assume);
I_per_res10=NLmeans_grey(I_per,sig_assume);
I_res10=(I_par_res10-I_per_res10)./(I_par_res10+I_per_res10);
I_res10(find(I_res10>1))=1; % delete the overflow
I_res10(find(I_res10<0))=0;
figure(6)
subplot(131)
imagesc(I_par_res10); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res10); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res10); colormap gray; axis off
title('I_{res}')
% sig=20
sig_assume=20;
I_par_res20=NLmeans_grey(I_par,sig_assume);
I_per_res20=NLmeans_grey(I_per,sig_assume);
I_res20=(I_par_res20-I_per_res20)./(I_par_res20+I_per_res20);
I_res20(find(I_res20>1))=1; % delete the overflow
I_res20(find(I_res20<0))=0;
figure(7)
subplot(131)
imagesc(I_par_res20); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res20); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res20); colormap gray; axis off
title('I_{res}')
% sig=30
sig_assume=30;
I_par_res30=NLmeans_grey(I_par,sig_assume);
I_per_res30=NLmeans_grey(I_per,sig_assume);
I_res30=(I_par_res30-I_per_res30)./(I_par_res30+I_per_res30);
I_res30(find(I_res30>1))=1; % delete the overflow
I_res30(find(I_res30<0))=0;
figure(8)
subplot(131)
imagesc(I_par_res30); colormap gray; axis off
title('I_{par}')
subplot(132)
imagesc(I_per_res30); colormap gray; axis off
title('I_{per}')
subplot(133)
imagesc(I_res30); colormap gray; axis off
title('I_{res}')
%% Calculate the contrast and variance for each parameter
sigma=[1 2 3 5 10 20 30];
% noisy iamge
I_polaziser=I_noisy(70:75,260:265);
I_background=I_noisy(95:100,270:275);
C_noisy=mean2(I_polaziser)-mean2(I_background);
V_noisy=std2(I_polaziser);
%sig=1
I_piece=OSC_noisy(20:100,220:300);
I_polaziser=OSC_noisy(70:75,260:265);
I_background=OSC_noisy(95:100,270:275);
figure()
imagesc(OSC_noisy);colormap gray;
figure()
imagesc(I_piece);colormap gray;
C(1)=mean2(I_polaziser)-mean2(I_background);
V(1)=std2(I_polaziser);
%sig=2
I_polaziser=I_res2(70:75,260:265);
I_background=I_res2(95:100,270:275);
C(2)=mean2(I_polaziser)-mean2(I_background);
%sig=3
I_polaziser=I_res3(70:75,260:265);
I_background=I_res3(95:100,270:275);
C(3)=mean2(I_polaziser)-mean2(I_background);
%sig=5
I_polaziser=I_res5(70:75,260:265);
I_background=I_res5(95:100,270:275);
C(4)=mean2(I_polaziser)-mean2(I_background);
%sig=10
I_polaziser=I_res10(70:75,260:265);
I_background=I_res10(95:100,270:275);
C(5)=mean2(I_polaziser)-mean2(I_background);
%sig=20
I_polaziser=I_res20(70:75,260:265);
I_background=I_res20(95:100,270:275);
C(6)=mean2(I_polaziser)-mean2(I_background);
%sig=30
I_polaziser=I_res30(70:75,260:265);
I_background=I_res30(95:100,270:275);
C(7)=mean2(I_polaziser)-mean2(I_background);
%% Establish a model to explain the blur phenomenon
clc
clear all
close all
% Dimensions
N = 256; 

% Parametrers

% Definition of the number of extimation iamges
nb_sectors = 4; % number of sectors 4*4
N_sec = N/nb_sectors; % number of pixels in each sector 64*64
size_target=8; % size of target 8*8

% Intensity of target and background(difference 20)
I_background=25;
I_target=5;

% Standard deviration of noise
sig=2;

% OSC
P=0.5; % OSC average
delta_P=0.2; % Difference of OSC
P_a = P + delta_P/2 ;  % OSC target
P_b = P - delta_P/2 ;  % OSC background

%%%%%%%%%%%%% Intensity set up
section=I_background*ones(N_sec,N_sec);
section(N_sec/2-size_target/2+1:N_sec/2+size_target/2,N_sec/2-size_target/2+1:N_sec/2+size_target/2)=I_target;
im_I=kron(ones(nb_sectors,nb_sectors),section);

%%%%%%%%%%%%% Polarization set up
cible=P_b*ones(N_sec,N_sec);
cible(N_sec/2-size_target/2+1:N_sec/2+size_target/2,N_sec/2-size_target/2+1:N_sec/2+size_target/2)=P_a;
im_cible = kron(ones(nb_sectors,nb_sectors),cible);

%%% Generation des images X et Y
X_original= im_I .* (1+im_cible)/2;
Y_original= im_I .* (1-im_cible)/2;
I_original=X_original+Y_original;
OSC_original=(X_original-Y_original)./(X_original+Y_original);

%%% Add noise to the OSC images
m=1;
for i=1:4
    for j=1:4
        X_noisy(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=X_original(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)+sig*randn(N_sec,N_sec);
        Y_noisy(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=Y_original(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)+sig*randn(N_sec,N_sec);
        m=m+1;
    end
end
I_noisy=X_noisy+Y_noisy;
OSC_noisy=(X_noisy-Y_noisy)./(X_noisy+Y_noisy);
OSC_noisy(find(OSC_noisy>1))=1; % delete the overflow
OSC_noisy(find(OSC_noisy<0))=0;
figure(1)
imagesc(I_noisy); colormap gray; axis off;
figure(2)
imagesc(OSC_noisy); colormap gray; axis off;
%% Implementation of NL-mean denoising
sig_assume =0.5:0.5:8;
m=1
for i=1:4
    for j=1:4
        X_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey(X_noisy(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig_assume(m));
        Y_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey(Y_noisy(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig_assume(m));
        m=m+1;
    end
end
I_new=X_new+Y_new;
OSC_new=(X_new-Y_new)./(X_new+Y_new);
OSC_new(find(OSC_new>1))=1;
OSC_new(find(OSC_new<0))=0;
figure(3)
imagesc(I_new); colormap gray; axis off;
figure(4)
imagesc(OSC_new); colormap gray; axis off;
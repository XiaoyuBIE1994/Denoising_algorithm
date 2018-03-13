%%% This profile is use to study the performance of NL-mean in simple OSC
%%% images
%% Initialization
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
%%% Dimensions
N = 256; 

%%% Parametres

% Definition du nombre de sous-images
nb_secteurs = 4;
N_sec = N/nb_secteurs;
taille_cible=N_sec/16;

% Intensite
I_min = 7; I_max = 50;

% Ecart type du bruit
sig =1;

% OSC
P=0.5; % OSC moyen
delta_P=0.2; % Difference d'OSC
P_a = P + delta_P/2 ;  % OSC cible
P_b = P - delta_P/2 ;  % OSC fond

% Demnesion  du filtre
d_filtre=3;


%%%%%%%%%%%%% Image secteurs

% Tableau des intensite
line_int = linspace(I_min,I_max,nb_secteurs^2);
tab_int = reshape (line_int,[nb_secteurs,nb_secteurs])';

% Image intensite
im_I=zeros(N,N);
for i=1:4
    for j=1:4
        im_I(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=tab_int (i,j);
    end
end
      
% Image cible
cible=P_b*ones(N_sec,N_sec);
cible(N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2,N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2)=P_a;
im_cible = kron(ones(nb_secteurs,nb_secteurs),cible);


%%% Generation des images X et Y
X_original= im_I .* (1+im_cible)/2;
Y_original= im_I .* (1-im_cible)/2;
X = im_I .* (1+im_cible)/2 + sig*randn(N,N);
Y = im_I .* (1-im_cible)/2 + sig*randn(N,N);
%%  Filtrage
tic
for i=1:4
    for j=1:4
        X_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey(X(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig);
        Y_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey(Y(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig);
    end
end
t=toc
%%  Pre-Filtering
% for i=1:4
%     for j=1:4
%         X_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey_gaussian(X(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig);
%         Y_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey_gaussian(Y(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig);
%     end
% end
%%  Pre-Filtering+NL-filter
% X_NL=X_new;
% Y_NL=Y_new;
% for i=1:4
%     for j=1:4
%         I_new=X_NL(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
%         X_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey_new(X(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig,I_new);
%         I_new=Y_NL(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
%         Y_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec)=NLmeans_grey_new(Y(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec),sig,I_new);
%     end
% end
%% Images polarimeriques 
I_im_original=X_original+Y_original;
P_im_original=(X_original-Y_original)./(X_original+Y_original);
I_im = X+Y;
P_im = (X-Y)./(X+Y);
I_im_new=X_new+Y_new;
P_im_new=(X_new-Y_new)./(X_new+Y_new);
figure(7); imagesc(linspace(I_min,I_max,N)*d_filtre/(sqrt(2)*sig),1:N,I_im); colormap gray;
figure(8); imagesc(linspace(I_min,I_max,N)*d_filtre/(sqrt(2)*sig),1:N,P_im); colormap gray;
figure(9); imagesc(linspace(I_min,I_max,N)*d_filtre/(sqrt(2)*sig),1:N,I_im_new); colormap gray;
figure(10); imagesc(linspace(I_min,I_max,N)*d_filtre/(sqrt(2)*sig),1:N,P_im_new); colormap gray;
%% ecart, cible et fond, bruite et restore, polarization
m=1;
M_fond=ones(N_sec);
M_fond(N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2,N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2)=0;
M_cible=ones(N_sec)-M_fond;
n=N_sec^2;
for i=1:4
    for j=1:4
        % three piece
        P_im_partie=P_im(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        P_im_partie_new=P_im_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        P_im_partie_original=P_im_original(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        % target
        P_im_cible=P_im_partie.*M_cible;
        P_im_cible_new=P_im_partie_new.*M_cible;
        P_im_cible_original=P_im_partie_original.*M_cible;
        % background
        P_im_fond=P_im_partie.*M_fond;
        P_im_fond_new=P_im_partie_new.*M_fond;
        P_im_fond_original=P_im_partie_original.*M_fond;
        mse_P_cible(m)=MSE_image(nonzeros(P_im_cible),nonzeros(P_im_cible_original));
        mse_P_cible_new(m)=MSE_image(nonzeros(P_im_cible_new),nonzeros(P_im_cible_original));
        mse_P_fond(m)=MSE_image(nonzeros(P_im_fond),nonzeros(P_im_fond_original));
        mse_P_fond_new(m)=MSE_image(nonzeros(P_im_fond_new),nonzeros(P_im_fond_original));
        m=m+1;
    end
end
figure(11)
plot(1:16,sqrt(mse_P_cible),'r',1:16,sqrt(mse_P_cible_new),'r--',1:16,sqrt(mse_P_fond),'b',1:16,sqrt(mse_P_fond_new),'b--')
legend('target','target new','background','background new')
xlabel('Intensity')
axis([1 16 0 0.3])
grid on
%% ecart, cible et fond, bruite et restore,X et Y
m=1;
M_fond=ones(N_sec);
M_fond(N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2,N_sec/2-taille_cible/2+1:N_sec/2+taille_cible/2)=0;
M_cible=ones(N_sec)-M_fond;
n=N_sec^2;
for i=1:4
    for j=1:4
        % partie X
        X_partie=X(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        X_partie_new=X_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        X_partie_original=X_original(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        % X target 
        X_cible=X_partie.*M_cible;
        X_cible_new=X_partie_new.*M_cible;
        X_cible_original=X_partie_original.*M_cible;
        % Y background
        X_fond=X_partie.*M_fond;
        X_fond_new=X_partie_new.*M_fond;
        X_fond_original=X_partie_original.*M_fond;
        % X MSE
        mse_X_cible(m)=MSE_image(nonzeros(X_cible),nonzeros(X_cible_original));
        mse_X_cible_new(m)=MSE_image(nonzeros(X_cible_new),nonzeros(X_cible_original));
        mse_X_fond(m)=MSE_image(nonzeros(X_fond),nonzeros(X_fond_original));
        mse_X_fond_new(m)=MSE_image(nonzeros(X_fond_new),nonzeros(X_fond_original));
        % partie Y
        Y_partie=Y(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        Y_partie_new=Y_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        Y_partie_original=Y_original(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        % Y target
        Y_cible=Y_partie.*M_cible;
        Y_cible_new=Y_partie_new.*M_cible;
        Y_cible_original=Y_partie_original.*M_cible;
        % Y background
        Y_fond=Y_partie.*M_fond;
        Y_fond_new=Y_partie_new.*M_fond;
        Y_fond_original=Y_partie_original.*M_fond;
        % Y MSE
        mse_Y_cible(m)=MSE_image(nonzeros(Y_cible),nonzeros(Y_cible_original));
        mse_Y_cible_new(m)=MSE_image(nonzeros(Y_cible_new),nonzeros(Y_cible_original));
        mse_Y_fond(m)=MSE_image(nonzeros(Y_fond),nonzeros(Y_fond_original));
        mse_Y_fond_new(m)=MSE_image(nonzeros(Y_fond_new),nonzeros(Y_fond_original));
        m=m+1;
    end
end
figure(12)
plot(1:16,sqrt(mse_X_cible),'r',1:16,sqrt(mse_Y_cible),'r--',1:16,sqrt(mse_X_fond),'b',1:16,sqrt(mse_Y_fond),'b--')
legend('X target','Y target','X background','Y background')
xlabel('Intensit¨¦')
grid on
figure(13)
plot(1:16,sqrt(mse_X_cible_new),'r',1:16,sqrt(mse_Y_cible_new),'r--',1:16,sqrt(mse_X_fond_new),'b',1:16,sqrt(mse_Y_fond_new),'b--')
legend('X target new','Y target new','X background new','Y background new')
xlabel('Intensity')
grid on
ratio_x=sum(sqrt(mse_X_fond))/sum(sqrt(mse_X_fond_new));
ratio_y=sum(sqrt(mse_Y_fond))/sum(sqrt(mse_Y_fond_new));
%% CRLB
sig_cible=(0.5*sqrt(mse_X_cible)+0.5*sqrt(mse_Y_cible)).^2;
sig_fond=(0.5*sqrt(mse_X_fond)+0.5*sqrt(mse_Y_fond)).^2;
sig_cible_new=(0.5*sqrt(mse_X_cible_new)+0.5*sqrt(mse_Y_cible_new)).^2;
sig_fond_new=(0.5*sqrt(mse_X_fond_new)+0.5*sqrt(mse_Y_fond_new)).^2;
CRLB_cible=(1+P_a^2)./((line_int.^2./(2*sig_cible)));
CRLB_fond=(1+P_b^2)./((line_int.^2./(2*sig_fond)));
CRLB_cible_new=(1+P_a^2)./((line_int.^2./(2*sig_cible_new)));
CRLB_fond_new=(1+P_b^2)./((line_int.^2./(2*sig_fond_new)));
figure(14)
plot(1:16,sqrt(CRLB_cible),'r',1:16,sqrt(CRLB_fond),'b',1:16,sqrt(CRLB_cible_new),'r--',1:16,sqrt(CRLB_fond_new),'b--')
legend('target','background','target new','background new')
xlabel('Intensity')
ylabel('CRLB')
axis([1 16 0 0.3])
grid on
%% 
m=1;
n=N_sec^2;
for i=1:4
    for j=1:4
        P_image(:,:,m)=P_im(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        P_image_new(:,:,m)=P_im_new(1+(i-1)*N_sec:i*N_sec,1+(j-1)*N_sec:j*N_sec);
        m=m+1;
    end
end
%%
figure(14)
subplot(241)
imagesc(P_image(:,:,13)); colormap gray;axis off;
subplot(242)
imagesc(P_image_new(:,:,13)); colormap gray;axis off;
subplot(243)
imagesc(P_image(:,:,14)); colormap gray;axis off;
subplot(244)
imagesc(P_image_new(:,:,14)); colormap gray;axis off;
subplot(245)
imagesc(P_image(:,:,15)); colormap gray;axis off;
subplot(246)
imagesc(P_image_new(:,:,15)); colormap gray;axis off;
subplot(247)
imagesc(P_image(:,:,16)); colormap gray;axis off;
subplot(248)
imagesc(P_image_new(:,:,16)); colormap gray;axis off;
%%
P_original=(X_original-Y_original)./(X_original+Y_original);
I_original=X_original+Y_original;
figure(15)
imagesc(I_original); colormap gray; axis off;
figure(16)
imagesc(P_original); colormap gray; axis off;
%% denoising research
close all
clc
k=5000; % times of simulation
for ds=1:5
Ds=5;
h=0.4*sig;
Value_original=X_original(15,15+N_sec)*ones(1,k);
Value_noisy=zeros(1,k);
Value_restore=zeros(1,k);
w=zeros(1,k);
r=zeros(1,k);
for p=1:k
X = im_I .* (1+im_cible)/2 + sig*randn(N,N);
I=X(1:N_sec,1+N_sec:2*N_sec);
I=double(I);
[m,n]=size(I);                                                                                                       
DenoisedImg=zeros(m,n);
PaddedImg = padarray(I(:,:),[ds,ds],'symmetric','both');
h2=h*h;
G=1/(2*ds+1)^2;
Value_noisy(p)=I(1,1);
for i=15:15
    for j=20:20
        i1=i+ds;% i1 and j1 are the position of the original image
        j1=j+ds;
        patch1=PaddedImg(i1-ds:i1+ds,j1-ds:j1+ds);%patch window 1
        sum_weight=0;
        NLmeans=0;
        % range of the researh window, limit of edge
        if (i1-Ds)>(ds+1) %transverse direction
            if(i1+Ds)<(m+ds)
                tmin=i1-Ds;
                tmax=i1+Ds;
            else
                tmin=m+ds-2*Ds;
                tmax=m+ds;
            end
        else
            tmin=ds+1;
            tmax=ds+1+2*Ds;
        end
        if (j1-Ds)>(ds+1) %vertical direction
            if(j1+Ds)<(n+ds)
                vmin=j1-Ds;
                vmax=j1+Ds;
            else
                vmin=n+ds-2*Ds;
                vmax=n+ds;
            end
        else
            vmin=ds+1;
            vmax=ds+1+2*Ds;
        end
        % scan in the research window
        m=1;
        for t=tmin:tmax
            for v=vmin:vmax
                d=sqrt((i1-t)^2+(j1-v)^2);
                patch2=PaddedImg(t-ds:t+ds,v-ds:v+ds);%patch window 2
                Euclidean2=G*(sum(sum((patch2-patch1).^2))); % Eucledean distance, 
                weight(m)=exp(-max(Euclidean2-2*sig^2,0)/h2); % weight
                NLmeans=NLmeans+weight(m)*PaddedImg(t,v);
                sum_weight=sum_weight+weight(m);
                m=m+1;
            end
        end
        DenoisedImg(i,j)=NLmeans/sum_weight;
        Value_restore(p)=DenoisedImg(i,j);
        a=size(find(weight==1));
        w(p)=a(2);
        r(p)=sum(weight.^2)/((sum(weight)).^2);
    end
end
end
Var_noisy=MSE_image(Value_noisy,Value_original);
Var_restore=MSE_image(Value_restore,Value_original);
ratio=sqrt(Var_noisy/Var_restore)
end
% figure()
% plot(1:p,w)
% figure()
% plot(1:(2*Ds+1)^2,weight,'x')
% figure()
% plot(1:p,r)
%%  plot
close all
a=[4.78 7.85 10.14 12.54 14.14 15.49 16.18 17.16 17.95 18.76];
b=[4.85 8.46 12.1 15.62 18.48 21.04 23.12 24.56 25.53 25.33];
c=[4.12 5.58 6.47 7.22 7.56];
figure()
plot(1:10,a,'--*',1:10,b,'--*',1:5,c,'--*')
xlabel('ds')
ylabel('ratio')
legend('Ds=10','Ds=15','Ds=5')
grid on
%% Ds=10
a=[4.78 7.85 10.14 12.54];
y=[4.63 7.06 8.16 8.66 ];
figure()
plot(1:4,a,'r--*',1:4,y,'k--*')
xlabel('ds')
ylabel('ratio')
legend('one pixel','Whole image')
grid on
%% Ds=5
c=[4.12 5.58 6.47 7.22];
y=[3.97 5.30 5.75 5.98];
figure()
plot(1:4,c,'b--*',1:4,y,'k--*')
xlabel('ds')
ylabel('ratio')
legend('one pixel','Whole image')
grid on
%% Ds=15
b=[4.85 8.46 12.1 15.62];
y=[4.79 7.36 8.96 10.55];
figure()
plot(1:4,b,'g--*',1:4,y,'k--*')
xlabel('ds')
ylabel('ratio')
legend('one pixel','Whole image')
grid on
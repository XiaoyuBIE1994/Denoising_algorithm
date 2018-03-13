function DenoisedImg=NLmeans_rgb_gaussian(I,ds,Ds,h,sigma)
%I:image with noise
%ds:size of patch window(2*ds+1)
%Ds:size of research window(2*Ds+1)
%h:filtering parameter
%DenoisedImg£ºimage after denoising
I=double(I);
[m,n,p]=size(I);                                                                                                       
DenoisedImg=zeros(m,n,p);
PaddedImg_r = padarray(I(:,:,1),[ds,ds],'symmetric','both');
PaddedImg_g = padarray(I(:,:,2),[ds,ds],'symmetric','both');
PaddedImg_b = padarray(I(:,:,3),[ds,ds],'symmetric','both');
tab_r=reshape(I(:,:,1),1,m*n);
tab_g=reshape(I(:,:,2),1,m*n);
tab_b=reshape(I(:,:,3),1,m*n);
Var_r=var(tab_r);
Var_g=var(tab_g);
Var_b=var(tab_b);
G_r=fspecial('gaussian',[2*ds+1 2*ds+1],Var_r);
G_g=fspecial('gaussian',[2*ds+1 2*ds+1],Var_g);
G_b=fspecial('gaussian',[2*ds+1 2*ds+1],Var_b);
h2=h*h;

for i=1:m
    for j=1:n
        i1=i+ds;% i1 and j1 are the position of the original image
        j1=j+ds;
        patch_r1=PaddedImg_r(i1-ds:i1+ds,j1-ds:j1+ds);%patch window 1 in red
        patch_g1=PaddedImg_g(i1-ds:i1+ds,j1-ds:j1+ds);%green
        patch_b1=PaddedImg_b(i1-ds:i1+ds,j1-ds:j1+ds);%blue
        sum_weight=0;
        NLmeans_r=0;
        NLmeans_g=0;
        NLmeans_b=0;
        % range of the researh window, limit of edge
        tmin = max(i1-Ds,ds+1); %transverse direction
        tmax = min(i1+Ds,m+ds);
        vmin = max(j1-Ds,ds+1); %vertical direction
        vmax = min(j1+Ds,n+ds);
        % scan in the research window
        for t=tmin:tmax
            for v=vmin:vmax
                d=sqrt((i1-t)^2+(j1-v)^2);
                patch_r2=PaddedImg_r(t-ds:t+ds,v-ds:v+ds);%patch window 2 in red
                patch_g2=PaddedImg_g(t-ds:t+ds,v-ds:v+ds);%in green
                patch_b2=PaddedImg_b(t-ds:t+ds,v-ds:v+ds);%in blue
                Euclidean2=1/3*(sum(sum(G_r.*(patch_r2-patch_r1).^2))+sum(sum(G_g.*(patch_g2-patch_g1).^2))+sum(sum(G_b.*(patch_b2-patch_b1).^2))); % Eucledean distance, 
                weight=exp(-max(Euclidean2-2*sigma^2,0)/h2); % weight
                NLmeans_r=NLmeans_r+weight*PaddedImg_r(t,v);
                NLmeans_g=NLmeans_g+weight*PaddedImg_g(t,v);
                NLmeans_b=NLmeans_b+weight*PaddedImg_b(t,v);
                sum_weight=sum_weight+weight;
            end
        end
        DenoisedImg(i,j,1)=NLmeans_r/sum_weight;
        DenoisedImg(i,j,2)=NLmeans_g/sum_weight;
        DenoisedImg(i,j,3)=NLmeans_b/sum_weight;
    end
end
end
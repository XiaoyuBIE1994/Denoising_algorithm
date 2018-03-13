%%% This is a NL-mean denoising algorithm for RGB image
function DenoisedImg=NLmeans_rgb(I,sigma)
%I:image with noise
%ds:size of patch window(2*ds+1)
%Ds:size of research window(2*Ds+1)
%h:filtering parameter
%DenoisedImg£ºimage after denoising
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
I=double(I);
[m,n,p]=size(I);                                                                                                       
DenoisedImg=zeros(m,n,p);
PaddedImg_r = padarray(I(:,:,1),[ds,ds],'symmetric','both');
PaddedImg_g = padarray(I(:,:,2),[ds,ds],'symmetric','both');
PaddedImg_b = padarray(I(:,:,3),[ds,ds],'symmetric','both');
G=1/(3*(2*ds+1)^2);
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
        for t=tmin:tmax
            for v=vmin:vmax
                d=sqrt((i1-t)^2+(j1-v)^2);
                patch_r2=PaddedImg_r(t-ds:t+ds,v-ds:v+ds);%patch window 2 in red
                patch_g2=PaddedImg_g(t-ds:t+ds,v-ds:v+ds);%in green
                patch_b2=PaddedImg_b(t-ds:t+ds,v-ds:v+ds);%in blue
                Euclidean2=G*(sum(sum((patch_r2-patch_r1).^2))+sum(sum((patch_g2-patch_g1).^2))+sum(sum((patch_b2-patch_b1).^2))); % Eucledean distance, 
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
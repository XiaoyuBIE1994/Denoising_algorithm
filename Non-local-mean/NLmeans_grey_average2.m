%%% This is a NL-mean denoising algorithm with a pre-filter(average with size of 2)
function DenoisedImg=NLmeans_grey_average2(I,sigma)
%I:image with noise
%ds:size of patch window(2*ds+1)
%Ds:size of research window(2*Ds+1)
%h:filtering parameter
%DenoisedImg£ºimage after denoising

if (sigma<=15)
    ds=1;
    Ds=10;
    h=0.4*sigma;
else if (sigma<=30)
        ds=2;
        Ds=10;
        h=0.4*sigma;
    else if (sigma<=45)
        ds=3;
        Ds=17;
        h=0.35*sigma;
        else if (sigma<=75)
            ds=4;
            Ds=17;
            h=0.35*sigma;
        else
            ds=5;
            Ds=17;
            h=0.3*sigma;
            end
        end
    end
end


I=double(I);
[m,n]=size(I);                                                                                                       
DenoisedImg=zeros(m,n);
PaddedImg = padarray(I(:,:),[ds,ds],'symmetric','both');
% classic filter
H=fspecial('average',2);
PaddedImg_new=imfilter(PaddedImg,H);
% NL-mean filter
% PaddedImg_new = padarray(I_new(:,:),[ds,ds],'symmetric','both');

h2=h*h;
G=1/(2*ds+1)^2;



for i=1:m
    for j=1:n
        i1=i+ds;% i1 and j1 are the position of the original image
        j1=j+ds;
        patch1=PaddedImg_new(i1-ds:i1+ds,j1-ds:j1+ds);%patch window 1
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
        for t=tmin:tmax
            for v=vmin:vmax
                d=sqrt((i1-t)^2+(j1-v)^2);
                patch2=PaddedImg_new(t-ds:t+ds,v-ds:v+ds);%patch window 2
                Euclidean2=G*(sum(sum((patch2-patch1).^2))); % Eucledean distance, 
                weight=exp(-max(Euclidean2-2*sigma^2,0)/h2); % weight
                NLmeans=NLmeans+weight*PaddedImg(t,v);
                sum_weight=sum_weight+weight;
            end
        end
        DenoisedImg(i,j)=NLmeans/sum_weight;
    end
end
end
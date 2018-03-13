function DenoisedImg=NLmeans(I,ds,Ds,h,sigma)
%I:image with noise
%ds:size of patch window(2*ds+1)
%Ds:size of research window(2*Ds+1)
%h:filtering parameter
%DenoisedImg£ºimage after denoising
I=double(I);
[m,n]=size(I);                                                                                                       
DenoisedImg=zeros(m,n);
PaddedImg = padarray(I(:,:),[ds,ds],'symmetric','both');
G=1/(2*ds+1)^2;
h2=h*h;

for i=1:m
    for j=1:n
        i1=i+ds;% i1 and j1 are the position of the original image
        j1=j+ds;
        patch1=PaddedImg(i1-ds:i1+ds,j1-ds:j1+ds);%patch window 1
        sum_weight=0;
        NLmeans=0;
        % range of the researh window, limit of edge
        tmin = max(i1-Ds,ds+1); %transverse direction
        tmax = min(i1+Ds,m+ds);
        vmin = max(j1-Ds,ds+1); %vertical direction
        vmax = min(j1+Ds,n+ds);
        % scan in the research window
        for t=tmin:tmax
            for v=vmin:vmax
                d=sqrt((i1-t)^2+(j1-v)^2);
                patch2=PaddedImg(t-ds:t+ds,v-ds:v+ds);%patch window 2
                Euclidean2=G*sum(sum((patch2-patch1).^2)); % Eucledean distance, 
                weight=exp(-max(Euclidean2-2*sigma^2,0)/h2); % weight
                NLmeans=NLmeans+weight*PaddedImg(t,v);
                sum_weight=sum_weight+weight;
            end
        end
        DenoisedImg(i,j)=NLmeans/sum_weight;
    end
end
end
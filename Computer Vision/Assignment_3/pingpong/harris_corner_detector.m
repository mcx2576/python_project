function [outputArg1,outputArg2,outputArg3] = harris_corner_detector(img)
%Create gaussian gradient filter
image=img;
k=5;
sigma=0.5;
G1=fspecial('gauss',[round(k*sigma),round(k*sigma)], sigma);
%Create the original mask
%[Gx, Gy]=gradient(G1);
dx = [1 0 -1; 2 0 -2; 1 0 -1];
dy=dx';
%Obtain Ix and Iy
Ix=conv2(img, dx, 'same');
Iy=conv2(img, dy, 'same');
%Convolute the Ix, Iy with the gaussian filter
Ix2 = conv2(Ix.^2, G1, 'same');
Iy2 = conv2(Iy.^2, G1, 'same');
Ixy = conv2(Ix.*Iy, G1,'same');
[rol,col]=size(Ix2);
H=zeros(rol,col);
for i = 1:rol
    for j = 1:col
        Q=[Ix2(i,j) Ixy(i,j); Ixy(i,j) Iy2(i,j)];
        %Calculate the eigen
        l=eig(Q);
        %Derive the H matrix
        H(i,j)=l(1)*l(2)-0.04*(l(1)+l(2))^2;
    end
end
%Compare with n*n window size neighbor
n=10;
%threshold = mean(mean(H))*5;
threshold = abs(mean2(H))*6.8;
count=1;
%Filter the original picture with local maximum
MX = ordfilt2(H,n^2,ones(n));
for i =5:rol
    for j = 5:col
        if (H(i,j)==MX(i, j)) && (H(i,j)>threshold)
            r(count)=i;
            c(count)=j;
            count = count + 1;
        end
    end
end
outputArg1 = H;
outputArg2 = r;
outputArg3 = c;
figure;
imshow(Ix);
figure
imshow(Iy);
figure
imshow(image);
hold on
radius=5;
t=0:0.1:2*pi;
for i=1:length(r)
    x_unit=c(i)+radius*sin(t);
    y_unit=r(i)+radius*cos(t);
    plot(x_unit,y_unit,'r');
end
hold off
end


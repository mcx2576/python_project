function [] =AWB(Image)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
a = im2double(imread(Image));
r = a(:,:,1);
g = a(:,:,2);
b = a(:,:,3);
avgR = mean(mean(r));
avgG = mean(mean(g));
avgB = mean(mean(b));
avgRGB = [avgR avgG avgB];
grayValue = (avgR + avgG + avgB)/3;
scaleValue = grayValue./avgRGB;
c(:,:,1)= scaleValue(1)*r; 
c(:,:,2)= scaleValue(2)*g; 
c(:,:,3)= scaleValue(3)*b; 
%   Display original picture and the refined picture in the same figure
figure(1)
subplot(1,2,1)
imshow(a)
subplot(1,2,2)
imshow(c)
end


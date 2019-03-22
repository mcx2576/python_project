function [outputArg1,outputArg2] = recolor(image1,image2,image3)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%   Read values from the original pictures
a_1 = im2double(imread(image2));
a = a_1;
b = im2double(imread(image3));
%   Recolor the picture with green by setting RGB channel(0,1,0) 
a(a(:,:,2)~=0) = 1;
a(a(:,:,1)~=0) = 0;
a(a(:,:,3)~=0) = 0;
c = a.*b;
figure(1)
subplot(1,2,1)
imshow(imread(image1))
subplot(1,2,2)
imshow(c)
%   Recolor the picture with green by setting RGB channel(0,1,0) 
a_1(a_1(:,:,1)~=0) = 1;
a_1(a_1(:,:,2)~=0) = 0;
a_1(a_1(:,:,3)~=0) = 1;
d=a_1.*b;
figure(2)
subplot(1,2,1)
imshow(imread(image1))
subplot(1,2,2)
imshow(d)
end


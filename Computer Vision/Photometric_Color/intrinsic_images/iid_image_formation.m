function [outputArg1,outputArg2] = reconstruct(image1,image2,image3)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
a = im2double(imread(image2));
b = im2double(imread(image3));
c = a.*b;
subplot(2,2,1)
imshow(imread(image1))
subplot(2,2,2)
imshow(a)
subplot(2,2,3)
imshow(b)
subplot(2,2,4)
imshow(c)
end


function [output_image] = rgb2grays(input_image)
% converts an RGB into grayscale by using 4 different methods
% Extrat values from R, G, B channels
im1 = im2double(input_image);
R = double(im1(:,:,1));
G = double(im1(:,:,2));
B = double(im1(:,:,3));
% ligtness method
%(max(R, G, B) + min(R, G, B)) / 2.
a = (max(max(R,G),B)+min(min(R,G),B))./2;
% average method
% (R + G + B) / 3
b = (R + G + B) ./ 3;
% luminosity method
% 0.21 R + 0.72 G + 0.07 B
c = 0.21 * R + 0.72 * G + 0.07 * B;
% built-in MATLAB function 
d = rgb2gray(im1);
dimension = size(input_image);
output_image = zeros(dimension(1),dimension(2),4);
output_image(:,:,1)=a;
output_image(:,:,2)=b;
output_image(:,:,3)=c;
output_image(:,:,4)=d;
end


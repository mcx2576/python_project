function [output_image] = rgb2normedrgb(input_image)
% converts an RGB image into normalized rgb
im1 = im2double(input_image);
R = double(im1(:,:,1));
G = double(im1(:,:,2));
B = double(im1(:,:,3));
r = R./(R+G+B);
g = G./(R+G+B);
b = B./(R+G+B);
im1(:,:,1) = r;
im1(:,:,2) = g;
im1(:,:,3) = b;
output_image = im1;
end


function [output_image] = rgb2opponent(input_image)
% converts an RGB image into opponent color space
im1 = im2double(input_image);
R = double(im1(:,:,1));
G = double(im1(:,:,2));
B = double(im1(:,:,3));
O_1 = (R-G)./sqrt(2);
O_2 = (R+G-2*B)./sqrt(6);
O_3 = (R+G+B)./sqrt(3);
im1(:,:,1) = O_1;
im1(:,:,2) = O_2;
im1(:,:,3) = O_3;
output_image = im1;
end


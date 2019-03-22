function [Gx, Gy, im_magnitude,im_direction] = compute_gradient(image)
image=im2double(image);

Sx = [1 0 -1; 2 0 -2; 1 0 -1];
Sy = [1 2 1; 0 0 0 ;-1 -2 -1];
Gx = imfilter(image, Sx, 'replicate', 'conv');
Gy = imfilter(image, Sy, 'replicate', 'conv');

im_magnitude = sqrt(Gx.^2 + Gy.^2);
im_direction = atan(Gy./Gx);

end


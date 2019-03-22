function [ albedo, normal ] = estimate_alb_nrm( image_stack, scriptV, shadow_trick)
%COMPUTE_SURFACE_GRADIENT compute the gradient of the surface
%   image_stack : the images of the desired surface stacked up on the 3rd
%   dimension
%   scriptV : matrix V (in the algorithm) of source and camera information
%   shadow_trick: (true/false) whether or not to use shadow trick in solving
%   	linear equations
%   albedo : the surface albedo
%   normal : the surface normal


%[h, w, ~] = size(image_stack);
[h, w, num_images] = size(image_stack);
if nargin == 2
    shadow_trick = true;
end

% create arrays for 
%   albedo (1 channel)
%   normal (3 channels)
albedo = zeros(h, w, 1);
normal = zeros(h, w, 3);

% =========================================================================
% YOUR CODE GOES HERE
% for each point in the image array
%   stack image values into a vector i
%   construct the diagonal matrix scriptI
%   solve scriptI * scriptV * g = scriptI * i to obtain g for this point
%   albedo at this point is |g|
%   normal at this point is g / |g|

% Go through each point (i,j) in the image
albedo = zeros(h,w);
normal = zeros(h,w,3);
for i = 1:h
    for j = 1:w
        % For this point x,y=(i,j), we calc the vector i_x_y
        % if shadowtrick=true, we have I*i=I*V*g, else just i=V*g
        % We solve I(x,y)*i(x,y)=I(x,y)*V*g(x,y) for g(x,y) by
        % mldivide, (which because it is "thin" uses QR to solve the least squares Ax=b)
        i_x_y = zeros(num_images,1);
        for k = 1:num_images
            i_x_y(k) = image_stack(i,j,k);
        end
        I_x_y = diag(i_x_y);
        
        % solving for g(x,y)
        warning off
        if sum(i_x_y) == 0
            g_x_y = zeros(3,1);        
        else
            if shadow_trick == true
                g_x_y = (I_x_y*scriptV)\(I_x_y*i_x_y);
            else
                g_x_y = scriptV\i_x_y;
            end
        end
        warning on
        
        % calculate albedo and normal for current point, and store it
        albedo(i,j) = norm(g_x_y);
        if albedo(i,j) ~= 0
            normal(i,j,:) = g_x_y/albedo(i,j);
        end
    end

end

% =========================================================================

end


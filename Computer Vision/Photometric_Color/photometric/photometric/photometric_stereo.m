close all
clear all
clc
 
disp('Part 1: Photometric Stereo')

assignment_rgb_part = true;
if assignment_rgb_part == false
    % obtain many images in a fixed view under different illumination
    disp('Loading images...')
    image_dir = './SphereGray5/';   % TODO: get the path of the script
    %image_dir = './MonkeyGray/'; 
    %image_ext = '*.png';

    [image_stack, scriptV] = load_syn_images(image_dir);
    [h, w, n] = size(image_stack);
    % part for selecting 10 out of 25 images of graysphere25
    %selected_indices = [1 2 3 4 5 6 7 8 9 10];
    %selected_indices = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25];
    %image_stack = image_stack(:,:,selected_indices);
    %scriptV = scriptV(selected_indices,:);
    %n=length(selected_indices);
    fprintf('Finish loading %d images.\n\n', n);
    
    % compute the surface gradient from the stack of imgs and light source mat
    disp('Computing surface albedo and normal map...')
    [albedo, normals] = estimate_alb_nrm(image_stack, scriptV,true);
else
    % obtain many images in a fixed view under different illumination
    disp('Loading images...')
    image_dir = './SphereColor/';   % TODO: get the path of the script
    %image_dir = './MonkeyColor/'; 
    %image_ext = '*.png';

    [image_stack_r, scriptV] = load_syn_images(image_dir,1);
    [image_stack_g, scriptV] = load_syn_images(image_dir,2);
    [image_stack_b, scriptV] = load_syn_images(image_dir,3);
    image_stack_r(isnan(image_stack_r)) = 0;
    image_stack_g(isnan(image_stack_g)) = 0;
    image_stack_b(isnan(image_stack_b)) = 0;
    image_stack = (1/3)*(image_stack_r + image_stack_g + image_stack_b);
    [h, w, n] = size(image_stack);

    % compute the surface gradient from the stack of imgs and light source mat
    disp('Computing surface albedo and normal map...')
    [~, normals] = estimate_alb_nrm(image_stack, scriptV,true);
    [albedo_R, ~] = estimate_alb_nrm(image_stack_r, scriptV,true);
    [albedo_G, ~] = estimate_alb_nrm(image_stack_g, scriptV,true);
    [albedo_B, ~] = estimate_alb_nrm(image_stack_b, scriptV,true);
    albedo = zeros(h,w,3);
    albedo(:,:,1) = albedo_R;
    albedo(:,:,2) = albedo_G;
    albedo(:,:,3) = albedo_B;
    
end



%% integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
disp('Integrability checking')
[p, q, SE] = check_integrability(normals);
%return%
threshold = 0.005;
%%%
%Extra part to say what to do when we discover an outlier:
% We replace it with 0
for i = 1:h
    for j = 1:w
        if SE(i,j) > threshold
            p(i,j) = 0;
            q(i,j) = 0;
        end
    end
end
%%%
SE(SE <= threshold) = NaN; % for good visualization
fprintf('Number of outliers: %d\n\n', sum(sum(SE > threshold)));

%% compute the surface height
height_map = construct_surface( p, q,'average' );

%% Display
show_results(albedo, normals, SE);
show_model(albedo, height_map);


%% Face
[image_stack, scriptV] = load_face_images('./yaleB02/');
[h, w, n] = size(image_stack);
% Remove bad images
indices_bad = [4,13,15,30,31,32,33,34,35,47,55,56,57,58,59,60,61,62,63,64];
image_stack(:,:,indices_bad) = [];
scriptV(indices_bad,:) = [];
n=n-length(indices_bad);
fprintf('Finish loading %d images.\n\n', n);
disp('Computing surface albedo and normal map...')
[albedo, normals] = estimate_alb_nrm(image_stack, scriptV, false);

%% integrability check: is (dp / dy  -  dq / dx) ^ 2 small everywhere?
disp('Integrability checking')
[p, q, SE] = check_integrability(normals);

threshold = 0.005;
%%%
%Extra part to say what to do when we discover an outlier:
%We replace it with 0
for i = 1:h
    for j = 1:w
        if SE(i,j) > threshold
            p(i,j) = 0;
            q(i,j) = 0;
        end
    end
end
%%%
SE(SE <= threshold) = NaN; % for good visualization
fprintf('Number of outliers: %d\n\n', sum(sum(SE > threshold)));

%% compute the surface height
height_map = construct_surface( p, q  ,'average');

show_results(albedo, normals, SE);
show_model(albedo, height_map);


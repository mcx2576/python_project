% Change flag demo to 'pingpong' for pingpong demo.
% Change flag demo to 'person_toy' for person_toy demo.

function demo_feature_tracking_new

demo = 'person_toy';%'pingpong';

if strcmp(demo,'pingpong') == true
    % Demo for pingpong samples:
    disp('Demo for pingpong samples');

    num_frames = 53;
    frames_orig = cell(1,num_frames);
    frames_gray = cell(1,num_frames);
    for k=1:num_frames
        frame_num = k-1;
        img_filename = sprintf('pingpong/%04d.jpeg', frame_num);
        img = imread(img_filename);
        frames_orig{k} = img;
        frames_gray{k} = im2double(rgb2gray(img));
    end
else
    % Demo for person_toy samples:
    disp('Demo for person_toy samples');

    num_frames = 104;
    frames_orig = cell(1,num_frames);
    frames_gray = cell(1,num_frames);
    for k=1:num_frames
        frame_num = k;
        img_filename = sprintf('person_toy/%08d.jpg', frame_num);
        img = imread(img_filename);
        frames_orig{k} = img;
        frames_gray{k} = im2double(rgb2gray(img));
    end
end

% Get feature points (corner points) of only first image
[~,r,c] = harris_corner_detector_no_plot(frames_gray{1});
corner_points = num2cell([c' r'], 2);
num_points = length(corner_points);
points_positions = cell(1,num_points);
for m = 1:num_points
    points_positions{m} = corner_points{m};
end
   
for k = 1:(num_frames-1)
    % Get 2 consecutive images
    img1 = frames_gray{k}; img2 = frames_gray{k+1};
    % Round point_positions to pixel points
    pixel_points = cell(1,num_points);
    for m = 1:num_points
        pixel_points{m}(1) = round(points_positions{m}(1));
        pixel_points{m}(2) = round(points_positions{m}(2));
    end
        
    % Calculate optical flow vectors for those feature points
    block_size = 15;
    opt_flow_vectors = calc_opt_flow_vectors(img1,img2,pixel_points,block_size);

    % Plot the optical flow vectors for those feature points on img1
    plot_opt_flow_vectors_on_img(frames_orig{k},pixel_points,opt_flow_vectors);

    % Update new positions of points
    for m = 1:num_points
        points_positions{m}(1) = points_positions{m}(1) + opt_flow_vectors{m}(1);
        points_positions{m}(2) = points_positions{m}(2) + opt_flow_vectors{m}(2);
    end
        
    % Save plot as frame for video
    video_frames(k) = getframe;
end
implay(video_frames);

end





function [Ix,Iy,It] = calc_Ix_Iy_It(img1,img2)
It = img1-img2;

sigma=2; kernel_size=9;
G_f = fspecial('gauss', kernel_size, sigma);
[Gx,Gy] = gradient(G_f);

Ix = imfilter(img1,Gx);
Iy = imfilter(img1,Gy);
end


function [regions] = get_regions(img, points, block_size)

num_points = length(points);
regions = cell(1, num_points);
num_pixels_per_region = block_size*block_size;
[h,w] = size(img);
reg_radius = (block_size-1)/2;

for k = 1:num_points
    % Store the pixels for the region corresponding with kth point
    reg_x_centre = points{k}(1); reg_y_centre = points{k}(2);
    if reg_x_centre <= reg_radius
        reg_x_centre = reg_radius + 1;
    elseif reg_x_centre >= (w-reg_radius)
        reg_x_centre = w-reg_radius-1;
    end
%    reg_x_centre = min(w-reg_radius-1, max(reg_radius+1, reg_x_centre));
    if reg_y_centre <= reg_radius
        reg_y_centre = reg_radius + 1;
    elseif reg_y_centre >= (h-reg_radius)
        reg_y_centre = h-reg_radius-1;
    end
%    reg_y_centre = min(h-reg_radius-1, max(reg_radius+1, reg_y_centre));

    reg_x_start = reg_x_centre - reg_radius; reg_x_end = reg_x_centre + reg_radius;
    reg_y_start = reg_y_centre - reg_radius; reg_y_end = reg_y_centre + reg_radius;
    regions{k} = img(reg_y_start:reg_y_end,reg_x_start:reg_x_end);
    regions{k} = reshape(regions{k}, [1,num_pixels_per_region]);
end

end

function [opt_flow_vectors] = calc_opt_flow_vectors(img1,img2,points,block_size)

num_regions = length(points);
opt_flow_vectors = cell(1, num_regions);

% Calculate Ix,Iy,It for whole image.
[Ix,Iy,It] = calc_Ix_Iy_It(img1,img2);

Ix_regions = get_regions(Ix, points, block_size);
Iy_regions = get_regions(Iy, points, block_size);
It_regions = get_regions(It, points, block_size);
warning off
for k = 1:num_regions
    [A,b] = calc_A_and_b(Ix_regions{k},Iy_regions{k},It_regions{k});
    v = inv(A' * A) * A' * b;
    opt_flow_vectors{k} = v;
end
warning on
end

function [A,b] = calc_A_and_b(reg_Ix, reg_Iy, reg_It)

num_pixels = length(reg_Ix);
A = zeros(num_pixels,2);
b = zeros(num_pixels,1);
for k = 1:num_pixels
    A(k,1) = reg_Ix(k);
    A(k,2) = reg_Iy(k);
    b(k) = -reg_It(k);
end

end

function plot_opt_flow_vectors_on_img(img, points, vectors)

num_points = length(points);
x_positions = zeros(1,num_points); y_positions = zeros(1,num_points);
vectors_x = zeros(1,num_points); vectors_y = zeros(1,num_points);
for k = 1:num_points
    x_positions(k) = points{k}(1); y_positions(k) = points{k}(2);
    vectors_x(k) = vectors{k}(1); vectors_y(k) = vectors{k}(2);
end
imshow(img);
hold on;
plot(x_positions,y_positions,'LineStyle','none','Color','r','Marker','o','MarkerSize',1);
quiver(x_positions, y_positions, vectors_x, vectors_y,'Color','r');
%quiver(x_positions, y_positions, vectors_x, vectors_y);

end





function [outputArg1,outputArg2,outputArg3] = harris_corner_detector_no_plot(img_gray)
k=5;
sigma=4;
G1=fspecial('gauss',[round(k*sigma),round(k*sigma)], sigma);
%Create the original mask
%[Gx, Gy]=gradient(G1);
dx = [1 0 -1; 2 0 -2; 1 0 -1];
dy=dx';
%Obtain Ix and Iy
%Ix=conv2(img, dx, 'same');
Ix = imfilter(img_gray,dx,'replicate');
Iy = imfilter(img_gray,dy,'replicate');
%Iy=conv2(img, dy, 'same');
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
threshold = abs(mean2(H))*20;
count=1;
%Filter the original picture with local maximum
MX = ordfilt2(H,n^2,ones(n));
for i =1:rol
    for j = 1:col
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
end


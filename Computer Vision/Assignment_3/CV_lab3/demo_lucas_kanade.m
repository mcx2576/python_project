% Change flag demo to 'sphere' for sphere demo.
% Change flag demo to 'synth' for synth demo.

function demo_lucas_kanade

demo = 'sphere';%'synth'
if strcmp(demo,'sphere') == true
    clf
    disp('Demo for sphere picture');
    img1_orig = imread('sphere1.ppm');
    img2_orig = imread('sphere2.ppm');
    img1 = im2double(rgb2gray(img1_orig));
    img2 = im2double(rgb2gray(img2_orig));
else
    clf
    disp('Demo for synth picture');
    img1_orig = imread('synth1.pgm');
    img2_orig = imread('synth2.pgm');
    img1 = im2double(img1_orig);
    img2 = im2double(img2_orig); 
end

block_size=15;
[h,w] = size(img1);
num_reg_hor = floor(w/block_size);
num_reg_ver = floor(h/block_size);
x_positions = (1:block_size:1+(num_reg_hor-1)*block_size) + (block_size-1)/2;
y_positions = (1:block_size:1+(num_reg_ver-1)*block_size) + (block_size-1)/2;

[X,Y] = meshgrid(x_positions,y_positions);
num_points = num_reg_hor*num_reg_ver;
points = cell(1,num_points);
X = reshape(X, [num_points,1]); Y = reshape(Y, [num_points,1]);
for k = 1:num_points
    points{k} = [X(k) Y(k)];
end

% Calculate optical flow vectors for those feature points
opt_flow_vectors = calc_opt_flow_vectors(img1,img2,points,block_size);

% Plot the optical flow vectors for those feature points on img1
plot_opt_flow_vectors_on_img(img1_orig,points,opt_flow_vectors);


end






function [Ix,Iy,It] = calc_Ix_Iy_It(img1,img2)
It = img1-img2;

sigma=2; kernel_size=round(6*sigma);
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
%plot(x_positions,y_positions,'LineStyle','none','Color','r','Marker','o','MarkerSize',1);
quiver(x_positions, y_positions, vectors_x, vectors_y,'Color','r');

end


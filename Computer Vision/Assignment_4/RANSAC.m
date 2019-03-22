function [optimal_params, o_n] = RANSAC(im1,im2)
% Implement the function created in Q1 to solve matches for images
[matches, scores] = keypoint_matching(im1,im2);
[~,a]=size(matches);

% Read the image and get the x and y coordinates
% read images
img1 = single(imread(im1));
img2 = single(imread(im2));
[f1, d1] = vl_sift(img1);
[f2, d2] = vl_sift(img2);
% Define the number of iteration
repeat = 30;
% Start the iteration and test
% Define the optimal number
o_n = 0;
optimal_params=0;
for k=1:repeat
% Random sample three matches for our two images.
    r = round((a-1).*rand(3,1) + 1);
    for i=1:3
        x_1(i) = f1(1,matches(1,r(i))) ;
        x_2(i) = f2(1,matches(2,r(i))) ;
        y_1(i) = f1(2,matches(1,r(i))) ;
        y_2(i) = f2(2,matches(2,r(i))) ;
    end
% Construct the matrix A of three pairs
    A=[x_1(1) y_1(1) 0 0 1 0;
        0 0 x_1(1) y_1(1) 0 1;
        x_1(2) y_1(2) 0 0 1 0;
        0 0 x_1(2) y_1(2) 0 1;
        x_1(3) y_1(3) 0 0 1 0;
        0 0 x_1(3) y_1(3) 0 1];
% Construct the b
    b=[x_2(1);y_2(1);x_2(2);y_2(2);x_2(3);y_2(3)];
% Calculate the parameters
    x = pinv(A)*b;
% Initialize a n store number of points
    n = 0;
% Recompute the transformed image using the current parameters
    for j=1:a
        point = [f1(1,matches(1,j)) f1(2,matches(1,j)) 0 0 1 0;
                  0 0 f1(1,matches(1,j)) f1(2,matches(1,j)) 0 1]* x; 
        %((point(1)-f2(1,matches(2,j)))^2 + (point(2)-f2(2,matches(2,j)))^2)
        if ((point(1)-f2(1,matches(2,j)))^2 + (point(2)-f2(2,matches(2,j)))^2)<100
             n=n+1;
        end
    end
% Compare n and optimal number
    if n > o_n
        optimal_params=x;
        o_n=n;
    end
end
% Calculate the x and y axis
% initialize 50 matches
r_a = round((a-1).*rand(50,1) + 1);
for k=1:50
    x1(k) = f1(1,matches(1,r_a(k))) ;   
    y1(k) = f1(2,matches(1,r_a(k))) ;
    p = [x1(k) y1(k) 0 0 1 0;0 0 x1(k) y1(k) 0 1]* optimal_params; 
    x2(k) = p(1) + size(img1,2) ;
    y2(k) = p(2) ;
end   



% Plot image side by side with transformed points
image1=imread(im1);
image2=imread(im2);
imshowpair(image1, image2, 'montage')
hold on
radius=5;
t=0:0.1:2*pi;
for j=1:50
    x_unit=x1(j)+radius*sin(t);
    y_unit=y1(j)+radius*cos(t);
    plot(x_unit,y_unit,'r');
    x_unit_2=x2(j)+radius*sin(t);
    y_unit_2=y2(j)+radius*cos(t);
    plot(x_unit_2,y_unit_2,'r');
end
     h = line([x1 ; x2], [y1 ; y2]) ;
    set(h,'linewidth', 1, 'color', 'b') ;
hold off
 
   
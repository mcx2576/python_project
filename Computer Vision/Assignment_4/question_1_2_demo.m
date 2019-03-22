[matches,score]=keypoint_matching('boat1.pgm','boat2.pgm');
[~,a]=size(matches);
%random generate 50 integers
r = round((a-1).*rand(50,1) + 1); 
%read images
img1 = imread('boat1.pgm');
img2 = imread('boat2.pgm');
im1 = single(img1);
im2 = single(img2);
[f1, d1] = vl_sift(im1);
[f2, d2] = vl_sift(im2);
for i=1:50
    x_1(i) = f1(1,matches(1,r(i))) ;
    x_2(i) = f2(1,matches(2,r(i))) + size(im1,2) ;
    y_1(i) = f1(2,matches(1,r(i))) ;
    y_2(i) = f2(2,matches(2,r(i))) ;
    if i >= 48
            x_1(i)
    x_2(i)
    y_1(i)
    y_2(i)
    end
end   
imshowpair(img1, img2, 'montage')
hold on
radius=5;
t=0:0.1:2*pi;
for i=48:50
    x_unit=x_1(i)+radius*sin(t);
    y_unit=y_1(i)+radius*cos(t);
    plot(x_unit,y_unit,'r');
    x_unit_2=x_2(i)+radius*sin(t);
    y_unit_2=y_2(i)+radius*cos(t);
    plot(x_unit_2,y_unit_2,'r');
end
    h = line([x_1 ; x_2], [y_1 ; y_2]) ;
    set(h,'linewidth', 1, 'color', 'b') ;
hold off
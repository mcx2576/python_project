 % Get the optimal parameters from RANSAC
[params, number]=RANSAC('boat2.pgm','boat1.pgm');
img2 = imread('boat2.pgm');
[row,col]=size(img2);
% Calculate the corner points of image one
%left-top
point_1=[1 1 0 0 1 0; 0 0 1 1 0 1]*params;
%right-top
point_2=[col 1 0 0 1 0; 0 0 col 1 0 1]*params;
%left-bottom
point_3=[1 row 0 0 1 0; 0 0 1 row 0 1]*params;
%right-bottom
point_4=[col row 0 0 1 0; 0 0 col row 0 1]*params;
%Calculate the max row and max col for transformed image
min_x=min([point_1(1) point_2(1) point_3(1) point_4(1)]);
max_x=max([point_1(1) point_2(1) point_3(1) point_4(1)]);
min_y=min([point_1(2) point_2(2) point_3(2) point_4(2)]);
max_y=max([point_1(2) point_2(2) point_3(2) point_4(2)]);
m_c=ceil(max_x-min_x);
m_r=ceil(max_y-min_y);

% initialize the matrix with zeros
t3=-min_x;
t4=-min_y;
u1=t3+params(5);
u2=t4+params(6);
T = [params(1) params(2) u1; params(3) params(4) u2; 0 0 1];
invT=inv(T);
A=zeros(m_r,m_c,'uint8');
for i=1:m_c
    for j=1:m_r
        % For pixel (i,j)
        coord_img1 = invT * [i;j;1];
        px=round(coord_img1(1));
        py=round(coord_img1(2));
        if px >= 1 && px <= col && py >= 1 && py<=row
            A(j,i) = img2(py,px);
        end
    end
end


%params is for img1 -> img2
% draw transformation of img1

% find x,y limits of pixels img1 in img2 coordinate system
figure(1)
imshow(A)
size(A)

%tform=affine2d([params(1) params(3) 0; params(2) params(4) 0; params(5) params(6) 1]);
tform = maketform('affine',[params(1) params(3) 0; params(2) params(4) 0; params(5) params(6) 1]);
figure(2)
B=imtransform(img2, tform, 'nearest');
size(B)
imshow(B)

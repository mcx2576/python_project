%Demo for Q1.3
img=im2double(imread('00000001.jpg'));
D=imrotate(img,45,'bilinear');
imwrite(D,'toy_rotate_45.jpg')
harris_corner_detector('toy_rotate_45.jpg')
A=imrotate(img,60,'bilinear');
imwrite(A,'toy_rotate_60.jpg')
harris_corner_detector('toy_rotate_60.jpg')
B=imrotate(img,90,'bilinear');
imwrite(B,'toy_rotate.jpg')
harris_corner_detector('toy_rotate.jpg')
C=imrotate(img,120,'bilinear');
imwrite(C,'toy_rotate_120.jpg')
harris_corner_detector('toy_rotate_120.jpg')
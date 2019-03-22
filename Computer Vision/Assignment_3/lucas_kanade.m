function [outputArg1,outputArg2] = lucas_kanade(synth,sphere)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[r,c]=size(synth);
r_1=floor(r/15);
c_1=floor(c/15);
picture_set=cell(1,r_1*c_1+1);
k=0;
for i = 1:r_1
    for j=1:c_1
        picture_set{k}=synth(((i-1)*15+1):i*15,((j-1)*15+1):j*15);
        k=k+1;
        if i==r_1 && j==c_1
            picture_set{k}=synth((r_1*15+1):r, (c_1*15+1):c);
        end
    end
end
for x =1:k
    v(x)=inv(picture_set{x}.'*picture_set{x})*picture_set{x}.'
end


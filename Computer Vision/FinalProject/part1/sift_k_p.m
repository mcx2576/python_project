function [d_a] = sift_k_p(x)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
if size(x,3)==3
    y = single(rgb2gray(x));
else
    y = single(x);
end 
[~, d_a] = vl_sift(y);
end


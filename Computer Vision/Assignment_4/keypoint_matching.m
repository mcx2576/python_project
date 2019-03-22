function [matches,scores] = keypoint_matching(img1,img2)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
img1 = single(imread(img1));
img2 = single(imread(img2));
[f1, d1] = vl_sift(img1);
[f2, d2] = vl_sift(img2);
[matches, scores] = vl_ubcmatch(d1, d2) ;
end


function [des] = sift(img,choice, binSize, magnif)
%function that imputs image and the choice and outputs descriptors
if choice == "gray"
des = sift_gray(img, binSize, magnif);
elseif choice == "RGBSIFT"
des = sift_RGB(img, binSize, magnif);
elseif choice == "rgbSIFT"
des = sift_nrgb(img, binSize, magnif);
elseif choice == "KeyPointSIFT"
des = sift_k_p(img);   
else 
des = sift_opponent(img, binSize, magnif);
end


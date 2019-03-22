function [d_a] = sift_gray(x, binSize, magnif)
%This function uses gray image extract sift features 
% Define the bin
if size(x,3)==3
    y = single(rgb2gray(x));
else
    y = single(x);
end 
I_a = vl_imsmooth(y, sqrt((binSize/magnif)^2 - .25)) ;
[~, d_a] = vl_phow(I_a, 'Sizes', binSize,'Step', 5, 'Color', 'gray');
end


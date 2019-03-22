function [d_a] = sift_opponent(x, binSize, magnif)
%Change the image color space
if size(x,3)==1
    y = cat(3, x, x, x);
    y = single(y);
else
    y = single(x);
end 
I_a = vl_imsmooth(y, sqrt((binSize/magnif)^2 - .25)) ;
[~, d_a] = vl_phow(I_a, 'Sizes', binSize,'Step', 5, 'Color', 'opponent');
end
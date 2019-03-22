function [d_a] = sift_nrgb(x, binSize, magnif)
%Change the image color space
if size(x,3)==1
    y = cat(3, x/3, x/3, x/3);
    y = single(y);
else
    R=x(:,:,1);
    G=x(:,:,2);
    B=x(:,:,3);
    r=R./(R+G+B);
    g=G./(R+G+B);
    b=B./(R+G+B);
    x=cat(3,r,g,b);
    y = single(x);
end 
I_a = vl_imsmooth(y, sqrt((binSize/magnif)^2 - .25)) ;
[~, d_a] = vl_phow(I_a, 'Sizes', binSize,'Step', 10, 'Color', 'rgb');
end
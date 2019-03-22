function [ PSNR ] = myPSNR( orig_image, approx_image )
I_max = double(max(max(orig_image)));
[h,w]=size(orig_image);
MSE=0;
for y=1:h
    for x=1:w
        MSE=MSE+(double(orig_image(y,x))-double(approx_image(y,x)))^2;
    end
end
MSE=MSE/(h*w);
RMSE=sqrt(MSE);
PSNR = 20*log10(I_max/RMSE);
end


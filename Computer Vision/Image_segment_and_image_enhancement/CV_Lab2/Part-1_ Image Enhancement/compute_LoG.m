function imOut = compute_LoG(image, LOG_type)
image=im2double(image);
switch LOG_type
    case 1
        %method 1
        sigma=0.5; kernel_size=5;
        gauss_f = fspecial('gaussian', kernel_size, sigma);
        laplac_f = fspecial('laplacian', 0);
        imOut = imfilter(image, gauss_f, 'replicate', 'conv');
        imOut = imfilter(imOut, laplac_f, 'replicate', 'conv');
    case 2
        %method 2
        sigma = 0.5; kernel_size=5;
        log_f = fspecial('log', kernel_size, sigma);
        imOut = imfilter(image, log_f, 'replicate', 'conv');
    case 3
        %method 3
        sigma_1 = 0.5; sigma_2 = 0.5*1.1; kernel_size = 5;
        gauss_f_1 = fspecial('gaussian', kernel_size, sigma_1);
        gauss_f_2 = fspecial('gaussian', kernel_size, sigma_2);
        dog_f = gauss_f_2 - gauss_f_1;
        imOut = imfilter(image, dog_f, 'replicate', 'conv');
end
end


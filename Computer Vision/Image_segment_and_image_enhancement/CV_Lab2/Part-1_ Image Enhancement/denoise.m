function [ imOut ] = denoise( image, kernel_type, varargin)
%image=im2double(image);
switch kernel_type
    case 'box' % varargin = kernel_size
        % For padding, we choose replicate
        kernel_size = varargin{1};
        imOut=imboxfilt(image, kernel_size, 'Padding', 'replicate');
    case 'median' % varargin = kernel_size
        % For padding, we choose zeros
        kernel_size = varargin{1};
        imOut = medfilt2(image, [kernel_size kernel_size], 'symmetric');%'zeros');
    case 'gaussian' % varargin = sigma, kernel_size
        % For padding, we choose replicate
        sigma = varargin{1}; kernel_size = varargin{2};
        g2d = gauss2D(sigma, kernel_size);
        imOut = imfilter(image, g2d, 'replicate', 'conv');
end

end

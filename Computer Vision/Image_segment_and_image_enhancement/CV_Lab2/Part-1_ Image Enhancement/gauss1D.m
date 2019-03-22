function G = gauss1D( sigma , kernel_size )
    G = zeros(1, kernel_size);
    if mod(kernel_size, 2) == 0
        error('kernel_size must be odd, otherwise the filter will not have a center to convolve on')
    end
    %% solution
    

    N=(kernel_size-1)/2;
    for n=1:kernel_size
        G(n)=(1 / (sigma*sqrt(2*pi))) * exp(-(n-N-1)^2 / (2*sigma^2));
    end
    % gives same result
    %G=(1 / (sigma*sqrt(2*pi))) * exp( -(-N:N).^2  / (2*sigma^2));
    
    %normalize
    G = G/sum(G);
    
end

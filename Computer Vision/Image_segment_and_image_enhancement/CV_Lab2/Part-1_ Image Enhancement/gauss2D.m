function G = gauss2D( sigma , kernel_size )
    %% solution
    g1d = gauss1D(sigma,kernel_size);
    
    for j=1:kernel_size
        for i=1:kernel_size
            G(i,j) = g1d(i)*g1d(j);
        end
    end
    G=G/sum(sum(G));
end

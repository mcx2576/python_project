function [ p, q, SE ] = check_integrability( normals )
%CHECK_INTEGRABILITY check the surface gradient is acceptable
%   normals: normal image
%   p : df / dx
%   q : df / dy
%   SE : Squared Errors of the 2 second derivatives

% initalization
[h,w,~] = size(normals);
p=zeros(h,w);%p = zeros(size(normals));
q=zeros(h,w);%q = zeros(size(normals));
SE=zeros(h,w);%SE = zeros(size(normals));

% ========================================================================
% YOUR CODE GOES HERE
% Compute p and q, where
% p measures value of df / dx
% q measures value of df / dy
[h,w,~] = size(normals);qq=0;
for i = 1:h
    for j = 1:w
        p(i,j) = normals(i,j,1)/normals(i,j,3);
        q(i,j) = normals(i,j,2)/normals(i,j,3);
    end
end
% ========================================================================



p(isnan(p)) = 0;
q(isnan(q)) = 0;



% ========================================================================
% YOUR CODE GOES HERE
% approximate second derivate by neighbor difference
% and compute the Squared Errors SE of the 2 second derivatives SE
for i = 2:(h-1)
    for j = 2:(w-1)
        dp_dy = -0.5*(p(i+1,j) - p(i-1,j));
        dq_dx = 0.5*(q(i,j+1) - q(i,j-1));
        SE(i,j) = (dp_dy-dq_dx)^2;
    end
end
% ========================================================================




end


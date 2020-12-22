function [mu,U,Y,sigma] = PCA_via_SVD(X, d)
%PCA_VIA_SVD Summary of this function goes here
%   Detailed explanation goes here
mu = mean(X,2);
X_normal = X;
[D, N] = size(X);
X_normal = X_normal - mu * ones([1,N]);
[U,sigma,~] = svd(X_normal);
Y = U(:,1:d)' * (X_normal);
end


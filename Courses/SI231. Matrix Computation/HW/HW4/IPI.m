function [lambda_list] = IPI(A, mu)
%P5 Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(A);
q = rand([n,1]);
lambda_list = [];
A_mu = A - mu * eye(n);
for i = 1:1:20
   z = inv(A_mu) *  q;
   q = z / norm(z);
   lambda_list = [lambda_list, q' * A * q];
end
end
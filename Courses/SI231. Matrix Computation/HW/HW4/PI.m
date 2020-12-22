function [lambda_list] = PI(A)
%P5 Summary of this function goes here
%   Detailed explanation goes here
[m,n] = size(A);
delta = 1;
q = rand([n,1]);
lambda_list = [];
for i = 1:1:20
   z = A * q;
   q = z / norm(z);
   lambda = q' * A * q;
   lambda_list = [lambda_list, lambda];
end
end


function [q]=GS2(A)
[m,n] = size(A);
q = zeros(m,n);
q = A;
for i=1:n
    q(:,i) = q(:,i)/norm(q(:,i));
    for j=i+1:n
        q(:,j) = q(:,j) - q(:,i)'* q(:,j) * q(:,i);
    end
end
end
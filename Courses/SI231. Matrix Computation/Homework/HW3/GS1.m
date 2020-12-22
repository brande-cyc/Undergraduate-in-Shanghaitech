function [q]=GS1(A)
[m,n] = size(A);
q = zeros(m,n);
q(:,1)=A(:,1)./norm(A(:,1));
for i=2:n
    vec(:,1) = A(:,i);
    for j=1:i-1
        delta = q(:,j)'*A(:,i)*q(:,j);
        vec = vec - delta;
    end
    q(:,i) = vec ./norm(vec);
end
end



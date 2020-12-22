function [R,y]=givens_matrix(x,i,j)
xi=x(i);          
xj=x(j);
r=sqrt(xi^2+xj^2);
cost=xi/r;
sint=-xj/r;
R=eye(length(x));
R(i,i)=cost;
R(i,j)=sint;
R(j,i)=-sint;
R(j,j)=cost;
y=x(:);
y([i,j])=[r,0];%sint cost


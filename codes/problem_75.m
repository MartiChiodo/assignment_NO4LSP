%% problem 75
clc
clear
close all

seed=339268;
rng(seed);

F_75= @(x) 0.5*((x(1)-1)^2 + sum( (10*(1:length(x)-1)'.*(x(2:end)-x(1:end-1)).^2).^2 ) );
gradF_75= @(x) gradient_pb_75(x);
hessF_75= @(x) hessian_pb_75(x);

%x=[-1;1;-1;1];
%F(x)
n=1e4;
x0= -1.2*ones(n,1);
x(n)= -1;
tic
[xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] ...
    = modified_Newton(F_75,gradF_75, hessF_75, x0, 50*n, [], [], [], [], [], [], ones(n,1));
time=toc

function grad = gradient_pb_75 (x)
n=length(x);
grad=zeros(n,1);
grad(1)= x(1)-1-200*(x(2)-x(1))^3;
%for k=2:n-1
%    grad(k)=200*((k-1)^2 * (x(k)-x(k-1))^3 - k^2 * (x(k+1)-x(k))^3 );
%end
grad(2:n-1)= 200*((1:n-2)'.^2 .* (x(2:n-1)-x(1:n-2)).^3 - (2:n-1)'.^2 .* (x(3:n)-x(2:n-1)).^3 );
grad(n)= 200*(n-1)^2*(x(n)-x(n-1))^3;
end

function hess = hessian_pb_75(x)
n=length(x);
diag_princ=zeros(n,1); %d^2F/dx_k^2
diag_princ(1)=1;
diag_princ(2:n-1)= 600* ((1:n-2)'.^2 .* (x(2:n-1) - x(1:n-2)).^2 + (2:n-1)'.^2 .* (x(3:n) - x(2:n-1)).^2 );
diag_princ(n)= 600 * (n-1)^2 * (x(n)-x(n-1))^2;
diag_upper=zeros(n,1);
diag_upper(2:n)= -600* ( (1:n-1)'.^2 .* (x(2:n)-x(1:n-1)).^2 );
diag_upper=diag_upper(2:n); %long n-1
%matrix whose columns are the diags of the sparse hessian
D=[[diag_upper;0], diag_princ, [0;diag_upper]];
hess=spdiags(D,-1:1,n,n);
end

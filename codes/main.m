
%% ESERCIZIO 2

clear all
clc


% Rosenbrock function in dimension n=2
f= @(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2; 
x0=[1.2;1.2];


function f = parametric_rosenbrock(x, alpha)
    f = 0;
    n = length(x);
    for i = 1:(n-1)
        f = f + alpha * (x(i+1) - x(i)^2)^2 + (1 - x(i))^2;
    end
end

function gradf = grad_parametric_rosenbrock(x,alpha)
    n = length(x);
    gradf = zeros(n,1);
    gradf(1,1) = -4*alpha*(x(2)-x(1)^2)*x(1) -4*(1-x(1)^2)*x(1);
    gradf(n,1) = 2*alpha*(x(n)-x(n-1)^2);

    for k = 2:n-1
        gradf(k, 1) = -4*alpha*(x(k+1)-x(k)^2)*x(k) -4*(1-x(k)^2)*x(k) + 2*alpha*(x(k)-x(k-1)^2);
    end
end

function Hessf = hess_parametric_rosenbrock(x,alpha)
    n = length(x);
    Hessf = zeros(n,n);

    for i = 1:n-1
        % Diagonal terms
        Hessf(i, i) = Hessf(i, i) + 2 - 4 * alpha * (x(i+1) - x(i)^2) + 12 * alpha * x(i)^2;
        Hessf(i+1, i+1) = Hessf(i+1, i+1) + 2 * alpha;
        
        % Off-diagonal terms
        Hessf(i, i+1) = Hessf(i, i+1) - 4 * alpha * x(i);
        Hessf(i+1, i) = Hessf(i+1, i) - 4 * alpha * x(i);
    end

end

f = @(x) parametric_rosenbrock(x, 100);
gradf = @(x) grad_parametric_rosenbrock(x,100);
Hessf = @(x) hess_parametric_rosenbrock(x,100);

[xbest_NM,xseq_NM,iter_NM,fbest_NM, flag, failure_NM] = nelderMead(f,x0,[],[],[],[],50,1e-9);
x0 = 2 * ones(1000,1);

itermax = 500; rho = 0.8; c1 = 1e-4; btmax = 50; tolgrad = 1e-9; tau_kmax = 10; 
[xbest_MN, xseq_MN, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] = modified_Newton(f,gradf, Hessf, x0, itermax, rho, c1, btmax, tolgrad, tau_kmax);

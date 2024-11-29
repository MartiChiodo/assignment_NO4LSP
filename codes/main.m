% ESERCIZIO 2

clear all
clc

% function that compute the rate of convergence
function rate_of_convergence = compute_roc(x_esatto, xseq)
if size(xseq,2) >=3
    rate_of_convergence = log(norm(x_esatto - xseq(:,end))/norm(x_esatto - xseq(:, end-1)))/log(norm(x_esatto - xseq(:,end-1))/norm(x_esatto - xseq(:, end-2)));
else 
    rate_of_convergence = nan;
end
end



% Parametric Rosenbrock function in dimension n 
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

% the excercice asks to fix alpha = 100
f = @(x) parametric_rosenbrock(x, 100);
gradf = @(x) grad_parametric_rosenbrock(x,100);
Hessf = @(x) hess_parametric_rosenbrock(x,100);

% initial points for the algorithms
x0_a = [1.2; 1.2];
x0_b = [-1.2; 1];
x_esatto = [1;1];

tol = 1e-8;
rho = 0.5; c1 = 1e-4; btmax = 50; tau_kmax = 10; 

time_SX = 0;
time_MN = 0;

% we run each model twice with different initial conditions and we compute
% some summary
t1 = tic;
[xbest_SX_a,xseq_SX_a,iter_SX_a,fbest_SX_a, flag_SX_a, failure_SX_a] = nelderMead(f,x0_a,[],[],[],[],400,tol);
time_SX = time_SX + toc(t1);
t1 = tic;
[xbest_MN_a, xseq_MN_a, iter_MN_a, fbest_MN_a, gradfk_norm_MN_a, btseq_MN_a, flag_acktrck_MN_a, failure_MN_a] = modified_Newton(f,gradf, Hessf, x0_a, 1000, rho, c1, btmax, tol, tau_kmax);
time_MN = time_MN + toc(t1);
t1 = tic;
[xbest_SX_b,xseq_SX_b,iter_SX_b,fbest_SX_b, flag_SX_b, failure_SX_b] = nelderMead(f,x0_b,[],[],[],[],400,tol);
time_SX = time_SX + toc(t1);
t1 = tic;
[xbest_MN_b, xseq_MN_b, iter_MN_b, fbest_MN_b, gradfk_norm_MN_b, btseq_MN_b, flag_bcktrck_MN_b, failure_MN_b] = modified_Newton(f,gradf, Hessf, x0_b, 1000, rho, c1, btmax, tol, tau_kmax);
time_MN = time_MN + toc(t1);

% creation of the table
failure = [failure_SX_a + failure_SX_b;  failure_MN_a + failure_MN_b];
avg_iter = [(iter_SX_b+iter_SX_a)/2; (iter_MN_b + iter_MN_a)/2];
avg_time_execution = [time_SX/2; time_MN/2];
roc = [(compute_roc(x_esatto, xseq_SX_b)+ compute_roc(x_esatto, xseq_SX_a))/2; (compute_roc(x_esatto, xseq_MN_b)+ compute_roc(x_esatto, xseq_MN_a))/2];


T = table( failure, avg_iter, avg_time_execution, roc, 'RowNames', {'simplex method'; 'modified Newton'});
format bank
display(T)



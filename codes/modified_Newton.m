function [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, itermax, rho, c1, btmax, tolgrad, tau_kmax)

% [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, itermax, rho, c1, btmax, tolgrad, tau_kmax)
%
% Function the minimizer of the function f by using the Modified Newton
% Method and implementing backtracking
% 
% INPUTS:
% f = function handle that return the values of function we want to minimize f_R^n --> R;
% gradf = function handle that compute the gradient of the function f in a given point;
% Hessf = function handle that compute the Hessian of the function f in a given point;
% x0 = starting point in R^n;
% itermax = maximum number of outter iterations;
% rho = ﻿fixed factor, lesser than 1, used for reducing alpha0;
% c1 = ﻿the factor of the Armijo condition that must be a scalar in (0,1);
% btmax =  ﻿maximum number of steps for updating alpha during the backtracking strategy;
% tolgrad = value used as stopping criterion w.r.t. the norm of the gradient;
% tau_kmax = maximum number of iterations permitted to compute Bk at each step;
% 
% OUTPUTS:
% xbest = the last xk computed by the function;
% xseq = matrix nx3 where we stored just the last 3 xk;
% iter = index of the last iteration performed;
% fbest = the value of f(xbest);
% gradfk_norm = value of the norm of gradf(xbest);
% btseq =  1-by-iter vector where elements are the number of backtracking iterations at each optimization step;
% flag_bcktrck = returns true if the method stopped because the backtracking failed;
% failure = returns true if the method stopped with iter = itermax and without satisfying the stopping criterion w.r.t- the norm of the gradient;
% 


% we are verifying that all the parameters are passed as inputs
if isempty(rho)
    rho=0.5;
end
if isempty(c1)
    c1 = 1e-4;
end
if isempty(itermax)
    itermax=500;
end
if isempty(tolgrad)
    tol=1e-6;
end
if isempty(btmax)
    btmax=40;
end

if isempty(tau_kmax)
    tau_kmax=50;
end



% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) fk + alpha * c1_gradfk_pk;

% initializing quantities
n = length(x0); %dimension
xseq = zeros(n,3);
cont = 1;
xseq(:,cont) = x0;
btseq = zeros(itermax,1);
failure = false;
flag_bcktrck = false;

fk = f(x0);
gradfk = gradf(x0);
Hessfk = Hessf(x0);
k = 0;

best_values = [];

while k < itermax && sum(gradfk.^2) > tolgrad^2
    
    % computing Bk according the Algorith 3.3 of the book
    beta = 1e-3;
    min_diag = min(diag(Hessfk));

    % initializing the tau
    if min_diag > 0
        tau_0 = 0;
    else
        tau_0 = -min_diag +beta;
    end

    for k_tau = 1:tau_kmax
        % attempt to compute the Choleski factorization
        failure_chol = false;
        try
            R = chol(Hessfk + tau_0 * spdiags(ones(n,1), 0, n, n));
            
        catch
            failure_chol = true;
        end

        if failure_chol
            tau_0 = max(5*tau_0, beta);
        else
            % we already solve the linear system to compute pk exploiting
            % the choleski factorization
            y = -  R' \ gradfk;
            pk = R \ y;
            break
        end
    end

    if k_tau == tau_kmax && failure_chol
        disp("ALGORITMO 3.3 HA FALLITO")
        xbest = 0; fbest = 0; iter = 0; gradfk_norm = 0;
        return
    end

    % BACKTRACKING
    % Reset the value of alpha
    alpha = 1;
    
    % Compute the candidate new xk
    xnew = x0 + alpha * pk;
    % Compute the value of f in the candidate new xk
    fnew = f(xnew);
    
    c1_gradfk_pk = c1 * gradfk' * pk;
    bt = 0;
    % Backtracking strategy: 
    % 2nd condition is the Armijo condition not satisfied
    while bt < btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        % Reduce the value of alpha
        alpha = rho * alpha;
        % Update xnew and fnew w.r.t. the reduced alpha
        xnew = x0 + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        flag_bcktrck = true;
        break
    end

    x0 = xnew;


    % preparing for the next iteration
    k = k+1;
    fk = f(x0);
    btseq(k,1) = bt;
    gradfk = gradf(x0);
    Hessfk = Hessf(x0);

    % updating xseq
    if cont == 3
        cont = 1;
    else
        cont = cont + 1;
    end
    xseq(:,cont) = x0;

    best_values(end+1) = fk;
    % plot
    figure(1);
    plot(best_values, '-o', 'MarkerSize', 4);
    xlabel('Iterations');
    ylabel('Best Evaluation');
    title('Progress minimum value Modified Newton Method');
    drawnow;

end

% declaring failure if this is the case
if k == itermax && sum(gradfk.^2) > tolgrad^2
    failure = true;
end

xbest = x0;
fbest = fk;
iter = k;
xseq = xseq(:,1:min(iter, 3));
btseq = btseq(1:iter,1);
gradfk_norm = norm(gradfk);

end
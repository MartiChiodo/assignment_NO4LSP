function [xk, fk, gradfk_norm, k, xseq, fseq, btseq, taoseq] = ...
    Modified_Newton_method(x0, f, gradf, Hessf, ...
    kmax, tolgrad, c1, rho, btmax)
%
% [xk, fk, gradfk_norm, k, xseq, btseq] = ...
    % newton_bcktrck(x0, f, gradf, Hessf, ...
    % kmax, tolgrad, c1, rho, btmax)
%
% Function that performs the Newton optimization method, using
% backtracking strategy for the step-length selection.
% 
% There is also the check in case of an Hessian not definite positive
%
% INPUTS:
% x0 = n-dimensional column vector;
% f = function handle that describes a function R^n->R;
% gradf = function handle that describes the gradient of f;
% Hessf = function handle that describes the Hessian of f;
% kmax = maximum number of iterations permitted;
% tolgrad = value used as stopping criterion w.r.t. the norm of the
% gradient;
% c1 = ﻿the factor of the Armijo condition that must be a scalar in (0,1);
% rho = ﻿fixed factor, lesser than 1, used for reducing alpha0;
% btmax = ﻿maximum number of steps for updating alpha during the 
% backtracking strategy.
%
% OUTPUTS:
% xk = the last x computed by the function;
% fk = the value f(xk);
% gradfk_norm = value of the norm of gradf(xk)
% k = index of the last iteration performed
% xseq = n-by-k matrix where the columns are the elements xk of the 
% sequence
% btseq = 1-by-k vector where elements are the number of backtracking
% iterations at each optimization step.
%

% Function handle for the armijo condition
farmijo = @(fk, alpha, c1_gradfk_pk) ...
    fk + alpha * c1_gradfk_pk;

% Initializations
xseq = zeros(length(x0), kmax);
fseq = zeros(1,kmax);
btseq = zeros(1, kmax);
taoseq = zeros(1,kmax);

xk = x0;
fk = f(xk);
gradfk = gradf(xk);
k = 0;
gradfk_norm = norm(gradfk);

best_values= [];
best_gradf = [];

while k < kmax && gradfk_norm >= tolgrad

    Hk = Hessf(xk);   % compute the Hessian
    tao = CholeskyAddIdentity(Hk);
    Bk = Hk + tao*diag(ones(length(x0),1)); % computation of Bk as a positive definite matrix


    % Compute the descent direction as solution of
    % Bk p = - gradf(xk)
    %%%%%% L.S. SOLVED WITH BACKSLASH (NOT USED) %%%%%%%%%%
    % pk = -Bk\gradfk;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%% L.S. SOLVED WITH pcg %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For simplicity: default values for tol and maxit; no preconditioning
    % pk = pcg(Bk, -gradfk);
    % If you want to silence the messages about "solution quality", use
    % instead: 
    % [pk, flagk, relresk, iterk, resveck] = pcg(Bk, -gradfk);
    [pk, ~, ~, iterk, ~] = pcg(Bk, -gradfk);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Reset the value of alpha
    alpha = 1;
    
    % Compute the candidate new xk
    xnew = xk + alpha * pk;
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
        xnew = xk + alpha * pk;
        fnew = f(xnew);
        
        % Increase the counter by one
        bt = bt + 1;
    end
    if bt == btmax && fnew > farmijo(fk, alpha, c1_gradfk_pk)
        break
    end
    
    % Update xk, fk, gradfk_norm
    xk = xnew;
    fk = fnew;
    gradfk = gradf(xk);
    gradfk_norm = norm(gradfk);
    
    % Increase the step by one
    k = k + 1;
    
    % Store current xk in xseq
    xseq(:, k) = xk;
    % Store current fk in fseq
    fseq(k) = fk;
    % Store bt iterations in btseq
    btseq(k) = bt;
    % Store current tao in taoseq
    taoseq(k) = tao;

    best_values(end+1) = fk;
    best_gradf(end+1) = gradfk_norm;
    if mod(k, 10) == 0 && k > 30
        figure(1);
        plot(best_values(25:end), '-o', 'MarkerSize', 4);
        xlabel('Iterations');
        ylabel('Best Evaluation');
        title('Progress minimum value Modified Newton Method');
        drawnow;

        figure(2);
        plot(best_gradf(25:end), '-o', 'MarkerSize', 4);
        xlabel('Iterations');
        ylabel('Best Evaluation');
        title('Progress gradient value Modified Newton Method');
        drawnow;

    end

end

% "Cut" xseq and btseq to the correct size
xseq = xseq(:, 1:k);
btseq = btseq(1:k);
fseq = fseq(1:k);
taoseq = taoseq(1:k);

% "Add" x0 at the beginning of xseq (otherwise the first el. is x1)
xseq = [x0, xseq];

end

function [tao] = CholeskyAddIdentity(Hessian)

    Beta = 10^-3;
    kmax = 1000;

    n = length(Hessian(1,:));
    
    % extract the min on the diagonal

    a = min(diag(Hessian));

    % compare

    if a > 0
        tao = 0;
    else
        tao = -a + Beta;
    end
        
    % computing A

    k = 1;
    while(k > 0)
        try
            R = chol(Hessian + tao *diag(ones(n,1)));
            k = -1;
        catch
            tao = max([2*tao;Beta]);
        end

        if k > kmax
            k = -1;
        end
    end


end
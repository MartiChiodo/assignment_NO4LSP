%% PROVE MODIFIED
%% prova modified cambiando parametri

f = @(x) parametric_rosenbrock(x, 100);
gradf = @(x) grad_parametric_rosenbrock(x,100);
Hessf = @(x) hess_parametric_rosenbrock(x,100);

tol = 1e-7;
n = 1e3;
x0 = ones(n,1); % pto iniziale Rosenbrock
x0(1:2:n) = -1.2;
rho = 0.7; c1 = 1e-1; btmax = 90; tau_kmax = 1e4; 
[xbest_MN, xseq_MN, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] = ...
    modified_Newton(f,gradf, Hessf, x0, 1000, rho, c1, btmax, tol, tau_kmax);

function f = parametric_rosenbrock(x, alpha)
    f = 0;
    n = length(x);
    for i = 2:n
        f = f + alpha * (x(i) - x(i-1)^2)^2 + (1 - x(i-1))^2;
    end
end

function gradf = grad_parametric_rosenbrock(x,alpha)
    n = length(x);
    gradf = zeros(n,1);
    
    for k = 2:n-1
        gradf(k,1) = -2*alpha*(x(k-1)^2 - x(k)) + 2*(x(k) -1) +4*alpha*x(k)*(x(k) - x(k+1));
    end

    gradf(1,1) = 2*(x(1) -1) + 4*alpha*x(1)*(x(1)^2 - x(2));
    gradf(n,1) = -2*alpha*(x(n-1)^2 - x(n)) ;

end

function Hessf = hess_parametric_rosenbrock(x,alpha)
    n = length(x);
    diags = zeros(n,3);
    % diags(:,1) is the principal one, diags(:,2) is the superior one and
    % diags(:,3) is the inferior one

    diags(1,1) = 2 + 8*alpha*x(1) - 4*alpha*x(2);
    diags(n,1) = 2*alpha;
    diags(n-1,3) = -4*alpha*x(n-1);
    diags(n,2) = -4*alpha*x(n-1);

    for k = 2:n-1
       diags(k,1) = 2*alpha*8*alpha*x(k) - 4*alpha*x(k+1) +2;
       diags(k-1,3) = -4*alpha*x(k-1); %diag inferior: k is the first derivative
       diags(k,2)= -4*alpha*x(k-1); %diag superior: k id the first derivative
    end
    

    Hessf = spdiags(diags, [0, +1, -1], n, n);

end

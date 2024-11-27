function [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, itermax, rho, c1, btmax, tolgrad, tau_kmax)

%%%% DESCRIZIONE INPUT/OUTPUT
% flag indica se il backtracking ha funzionato 
% tolgrad è usato come stopping criterion per il gradiente
% tau_kmax max iter per determinare Bk


% we are verifying that all the parameters are passed as inputs, eventually
% we set rho, chi, gamma and sigma with default valuess
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
xseq = zeros(n,itermax + 1);
xseq(:,1) = x0;
btseq = zeros(itermax,1);
failure = false;
flag_bcktrck = false;

fk = f(x0);
gradfk = gradf(x0);
Hessfk = Hessf(x0);
k = 0;

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
            R = chol(Hessfk + tau_0 * eye(n));
            
        catch
            failure_chol = true;
        end

        if failure_chol
            tau = max(5*tau_0, beta);
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
    xseq(:,k+1) = x0;
    btseq(k,1) = bt;
    gradfk = gradf(x0);
    Hessfk = Hessf(x0);

end

% declaring failure if this is the case
if k == itermax && sum(gradfk.^2) > tolgrad^2
    failure = true;
end

xbest = x0;
fbest = fk;
iter = k;
xseq = xseq(:, 1:iter+1);
btseq = btseq(1:iter,1);
gradfk_norm = norm(gradfk);

end
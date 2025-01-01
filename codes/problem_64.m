%% PROBLEMA 64
% non capisco dove abbia minimo
close all
clear all
clc

% setting the seed
seed = min(339268, 343310); 

% function to compute the rate of convergence
function rate_of_convergence = compute_roc(xseq)
if size(xseq,2) >=4
    k = size(xseq,2) -1;
    norm_ekplus1 = norm(xseq(:, k+1) - xseq(:,k));
    norm_ek = norm(xseq(:, k) - xseq(:,k-1));
    norm_ekminus1 = norm(xseq(:, k-1) - xseq(:,k-2));
    rate_of_convergence = log(norm_ekplus1/norm_ek) / log(norm_ek/norm_ekminus1);
else 
    rate_of_convergence = nan;
end
end


% implementing the function, the gradient and the hessiano for problem 64
function val = function_pb76(x)
    n = length(x);
    rho = 10;
    h = 1/(n+1);

    val = 0.5*(2*x(1) + rho*h^2* sinh(rho*x(1)) -x(2))^2 + 0.5*(2*x(n) + rho*h^2* sinh(rho*x(n)) -x(n-1) -1)^2;
    for k = 2:n-1
        val = val + 0.5*(2*x(k) + rho*h^2* sinh(rho*x(k)) -x(k-1) -x(k+1))^2;
    end
    

end

f = @(x) function_pb76(x);

function grad = grad_pb76(x)
    n = length(x);
    rho = 10;
    h = 1/(n+1);

    grad = zeros(n, 1);
    if n>=4
        grad(1,1) = (2*x(1) + rho*h^2*sinh(rho*x(1)) - x(2)) * (2 + rho^2*h^2*cosh(rho*x(1))) - (2*x(2) +  rho*h^2*sinh(rho*x(2)) - x(1) -x(3));
        grad(2,1) = (2 + rho^2*h^2*cosh(rho*x(2)))*(2*x(2) + rho*h^2*sinh(rho*x(2)) -x(1) -x(3)) - (2*x(1) +  rho*h^2*sinh(rho*x(1)) -x(2)) - (2*x(3) +  rho*h^2*sinh(rho*x(3)) - x(2) -x(4));
        grad(n-1,1) = (2 + rho^2*h^2*cosh(rho*x(n-1)))*(2*x(n-1) + rho*h^2*sinh(rho*x(n-1)) -x(n-2) -x(n)) - (2*x(n-2) +  rho*h^2*sinh(rho*x(n-2)) - x(n-3) -x(n-1)) - (2*x(n) +  rho*h^2*sinh(rho*x(n)) - x(n-1) - 1);
        for k =3:n-2
            grad(k,1) = (2 + rho^2*h^2*cosh(rho*x(k)))*(2*x(k) + rho*h^2*sinh(rho*x(k)) -x(k-1) -x(k+1)) - (2*x(k-1) +  rho*h^2*sinh(rho*x(k-1)) - x(k-2) -x(k)) - (2*x(k+1) +  rho*h^2*sinh(rho*x(k+1)) - x(k) -x(k+2));
        end
    elseif n == 2
        grad(1,1) = (2*x(1) + rho*h^2*sinh(rho*x(1)) - x(2)) * (2 + rho^2*h^2*cosh(rho*x(1))) - (2*x(2) +  rho*h^2*sinh(rho*x(2)) - x(1) -1); 
    elseif n == 3
        grad(1,1) = (2*x(1) + rho*h^2*sinh(rho*x(1)) - x(2)) * (2 + rho^2*h^2*cosh(rho*x(1))) - (2*x(2) +  rho*h^2*sinh(rho*x(2)) - x(1) -x(3));
        grad(2,1) = (2 + rho^2*h^2*cosh(rho*x(2)))*(2*x(2) + rho*h^2*sinh(rho*x(2)) -x(1) -x(3)) - (2*x(1) +  rho*h^2*sinh(rho*x(1)) -x(2)) - (2*x(3) +  rho*h^2*sinh(rho*x(3)) - x(2) - 1);
    end
    grad(n, 1) = -(2*x(n-1) + rho*h^2*sinh(rho*x(n-1)) - x(n-1) -x(n)) + (2 + rho^2*h^2*cosh(rho*x(n)))*(2*x(n) + rho*h^2*sinh(rho*x(n)) -x(n-1) - 1);

end

gradf = @(x) grad_pb76(x);


function val = hessian_pb76(x)
    n = length(x);
    rho = 10;
    h = 1/(n+1);
    diags = zeros(n,5); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior

    % principal diag
    diags(2:n-1,1) = (2+rho^2 * h^2*cosh(rho*x(2:n-1))).^2 + (2*x(2:n-1) + rho*h^2*sinh(rho*x(2:n-1)) - x(1:n-2) - x(3:n))*rho^3*h^2.*sinh(rho*x(2:n-1)) + 2;
    diags(1,1) =  (2+rho^2 * h^2*cosh(rho*x(1))).^2 + (2*x(1) + rho*h^2*sinh(rho*x(1)) - x(2))*rho^3*h^2*sinh(rho*x(1)) + 1;
    diags(n,1) = (2+rho^2 * h^2*cosh(rho*x(n))).^2 + (2*x(n) + rho*h^2*sinh(rho*x(n)) - x(n-1) - 1)*rho^3*h^2*sinh(rho*x(n)) + 1;

    % inferior diagonal
    diags(1:n-1,3) = -4 - rho^2*h^2*cosh(rho*x(1:n-1)) -  rho^2*h^2*cosh(rho*x(2:n));

    %superior diagonal
    diags(2:n,2) =   -4 - rho^2*h^2*cosh(rho*x(1:n-1)) -  rho^2*h^2*cosh(rho*x(2:n));

    % 2-inf e 2-suo diag
    diags(1:n-2, 5) = ones(n-2,1);
    diags(2:n, 4) = ones(n-1,1);

    val = spdiags(diags, [0,1,-1, 2, -2], n,n);
end


Hessf = @(x) hessian_pb76(x);

tol = 1e-4;


%% RUNNING THE EXPERIMENTS ON NEALDER MEAD
format short e
clc

% setting the dimensionality
dimension = [10 25 50];
iter_max = 400;
rng(seed);


% initializing the structures to store some stats
execution_time_SX = zeros(length(dimension),11);
failure_struct_SX = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_SX = zeros(length(dimension),11);
fbest_struct_SX = zeros(length(dimension),11);
roc_struct_SX = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);

    % defining the given initial point
    x0 = ones(n,1);

    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);

    % SOLVING SIMPLEX METHOD
    % first initial point
    t1 = tic;
    [~, xseq,iter,fbest, ~, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    execution_time_SX(dim,1) = toc(t1);
    fbest_struct_SX(dim,1) = fbest;
    iter_struct_SX(dim,1) = iter;
    roc_struct_SX(dim,1) = compute_roc(xseq);
    disp(['**** SIMPLEX METHOD FOR THE PB 64 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

    disp(['Time: ', num2str(execution_time_SX(dim,1)), ' seconds']);

    disp('**** SIMPLES METHOD : RESULTS *****')
    disp('************************************')
    disp(['f(xk): ', num2str(fbest)])
    disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
    disp(['Rate of Convergence: ', num2str(roc_struct_SX(dim,1))])
    disp('************************************')

    if (failure)
        disp('FAIL')
        disp('************************************')
    else
        disp('SUCCESS')
        disp('************************************')
    end
    disp(' ')


    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_SX(dim,1) = failure_struct_SX(dim,1) + failure;

    for i = 1:10
        t1 = tic;
        [~,~,iter,fbest, ~, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        execution_time_SX(dim,i+1) = toc(t1);
        fbest_struct_SX(dim,i+1) = fbest;
        iter_struct_SX(dim,i+1) = iter;
        failure_struct_SX(dim,i+1) = failure_struct_SX(dim,i+1) + failure;
        roc_struct_SX(dim,i+1) = compute_roc(xseq);

        disp(['**** SIMPLEX METHOD FOR THE PB 64 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_SX(dim,1)), ' seconds']);
    
        disp('**** SIMPLES METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
        disp(['Rate of Convergence: ', num2str(roc_struct_SX(dim,1))])
        disp('************************************')
    
        if (failure)
            disp('FAIL')
            disp('************************************')
        else
            disp('SUCCESS')
            disp('************************************')
        end
        disp(' ')

    end
end


varNames = ["average fbest", "average number of iterations", "average time of execution (sec)", "numbers of failure", "average rate of convergence"];
rowNames = string(dimension');
TSX = table(sum(fbest_struct_SX,2)/11, sum(iter_struct_SX,2)/11, sum(execution_time_SX,2)/11, sum(failure_struct_SX,2),sum(roc_struct_SX,2)/11 ,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TSX)



%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD
format short e
clc

% setting the values for the dimension
dimension = [1e3 1e4 1e5];
iter_max = 15000;

param = [0.4, 1e-4, 38; 0.4, 1e-4, 38; 0.4, 1e-4, 38;];

rng(seed);

% initializing structures to store some stats
execution_time_MN = zeros(length(dimension),11);
failure_struct_MN = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_MN = zeros(length(dimension),11);
fbest_struct_MN = zeros(length(dimension),11);
gradf_struct_MN = zeros(length(dimension),11);
roc_struct_MN = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);

    [rho, c1, btmax] = deal(param(dim, 1), param(dim, 2), param(dim, 3));


    %defining the given initial point
    x0 = ones(n,1);
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
    execution_time_MN(dim,1) = toc(t1);
    fbest_struct_MN(dim,1) = fbest;
    iter_struct_MN(dim,1) = iter;
    gradf_struct_MN(dim,1) = gradfk_norm;
    roc_struct_MN(dim,1) = compute_roc(xseq);
    disp(['**** MODIFIED NEWTON METHOD FOR THE PB 64 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

    disp(['Time: ', num2str(execution_time_MN(dim,1)), ' seconds']);
    disp(['Backtracking parameters (rho, c1): ', num2str(rho), ' ', num2str(c1)]);

    disp('**** MODIFIED NEWTON METHOD : RESULTS *****')
    disp('************************************')
    disp(['f(xk): ', num2str(fbest)])
    disp(['norma di gradf(xk): ', num2str(gradfk_norm)])
    disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max)])
    disp(['Rate of Convergence: ', num2str(roc_struct_MN(dim,1))])
    disp('************************************')

    if (failure)
        disp('FAIL')
        if (flag_bcktrck)
            disp('Failure due to backtracking')
        else
            disp('Failure not due to backtracking')
        end
        disp('************************************')
    else
        disp('SUCCESS')
        disp('************************************')
    end
    disp(' ')

    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_MN(dim,1) = failure_struct_MN(dim,1) + failure ;

    for i = 1:10
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
        execution_time_MN(dim,i+1) = toc(t1);
        fbest_struct_MN(dim,i+1) = fbest;
        iter_struct_MN(dim,i+1) = iter;
        failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
        gradf_struct_MN(dim,i+1) = gradfk_norm;
        roc_struct_MN(dim,i+1) = compute_roc(xseq);

        disp(['**** MODIFIED NEWTON METHOD FOR THE PB 64 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_MN(dim,1)), ' seconds']);
        disp(['Backtracking parameters (rho, c1): ', num2str(rho), ' ', num2str(c1)]);
    
        disp('**** MODIFIED NEWTON METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['norma di gradf(xk): ', num2str(gradfk_norm)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max)])
        disp(['Rate of Convergence: ', num2str(roc_struct_MN(dim,1))])
        disp('************************************')
    
        if (failure)
            disp('FAIL')
            if (flag_bcktrck)
                disp('Failure due to backtracking')
            else
                disp('Failure not due to backtracking')
            end
            disp('************************************')
        else
            disp('SUCCESS')
            disp('************************************')
        end
        disp(' ')
    end
end


varNames = ["avg fbest", "avg gradf_norm","avg num of iters", "avg time of exec (sec)", "n failure", "avg roc"];
rowNames = string(dimension');
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TMN)




%% FINITE DIFFERENCES
clc

function grad_approx = findiff_grad_64(x, h)
n = length(x);
    grad_approx = zeros(n,1);
    rho = 10;
    cost = 1/(n+1);

    fk_quadro = @(x,k) (2*x(k) + rho*cost^2 *sinh(x(k)) - x(k-1) - x(k+1))^2;

    eh_1 = zeros(n,1);
    eh_1(1) = h;
    eh_n = zeros(n,1);
    eh_n(n) = h;
    eh_2 = zeros(n,1);
    eh_2(2) = h;
    eh_n1 = zeros(n,1);
    eh_n1(n-1) = h;

    grad_approx(1,1) = ((2*(x(1)+h) + rho*cost^2 * sinh(x(1)+h) -x(2))^2 - (2*(x(1)-h) + rho*cost^2 * sinh(x(1)-h) - x(2))^2 + fk_quadro(x + eh_1, 2) - fk_quadro(x - eh_1, 2)) /(2*h);
    grad_approx(n,1) =  ((2*(x(n)+h) + rho*cost^2 * sinh(x(n)+h) -x(n-1) -1)^2 - (2*(x(n)-h) + rho*cost^2 * sinh(x(n)-h) - x(n-1) -1)^2 + fk_quadro(x + eh_n, n-1) - fk_quadro(x - eh_n, n-1)) /(2*h);
    grad_approx(2,1) = ((2*(x(1)) + rho*cost^2 * sinh(x(1)) -x(2) -h)^2 - (2*(x(1)) + rho*cost^2 * sinh(x(1)) - x(2) +h)^2 + fk_quadro(x + eh_2, 2) - fk_quadro(x - eh_2, 2) + fk_quadro(x + eh_2, 3) - fk_quadro(x - eh_2, 3)) /(2*h);
    grad_approx(n-1,1) =  ((2*(x(n)) + rho*cost^2 * sinh(x(n)) -x(n-1) -h -1)^2 - (2*(x(n)) + rho*cost^2 * sinh(x(n)) - x(n-1) +h -1)^2 + fk_quadro(x + eh_n1, n-1) - fk_quadro(x - eh_n1, n-1) + fk_quadro(x + eh_n1, n-2) - fk_quadro(x - eh_n1, n-2)) /(2*h);

    for k = 3:n-2
            eh_k = zeros(n,1);
            eh_k(k) = h;
        grad_approx(k,1) = (fk_quadro(x + eh_k,k) -fk_quadro(x-eh_k, k) +fk_quadro(x + eh_k,k-1) -fk_quadro(x-eh_k, k-1) +fk_quadro(x + eh_k,k+1) -fk_quadro(x-eh_k, k+1) )/(2*h);
    end
end
% 
% 
% function hessian_approx = findiff_hess_64(x, h)
%     % Calcola la matrice Hessiana sparsa per la funzione f(x)
%     % Input:
%     %   - x: vettore colonna (punto in cui calcolare l'Hessiana)
%     %   - h: passo
%     % Output:
%     %   - H: matrice Hessiana sparsa
% 
%     n = length(x); % Dimensione del problema
% 
%     fk_quadro = @(x,k) (x(k) - x(k+1)^2/10)^2;
%     fn_quadro = @(x) (x(n) - x(1)^2/10)^2;
% 
%     % Preallocazione per la struttura sparsa
%     i_indices = zeros(3*n,1);
%     j_indices = zeros(3*n,1);
%     values = zeros(3*n,1);
%     cont = 1;
% 
%     % Loop su k (dalla definizione della funzione)
%     for k = 1:n
%         % defyning the perturbation
%         he_k = zeros(n,1);
%         he_k(k) = h;
% 
%         % Elementi diagonali H(k, k)
%         if k == 1
%             H_kk = (fn_quadro(x+2*he_k) + fk_quadro(x+2*he_k, 1) -  2*fn_quadro(x+he_k) -2*fk_quadro(x+he_k,1) + fn_quadro(x) + fk_quadro(x,1))/(2*h^2);
%             i_indices(cont) = k;
%             j_indices(cont) = k;
%             values(cont) = H_kk;
%             cont = cont +1;
%         elseif k < n
%             H_kk = (fk_quadro(x+2*he_k, k-1) + fk_quadro(x+2*he_k, k) -  2*fk_quadro(x+he_k, k-1) -2*fk_quadro(x+he_k,k) + fk_quadro(x,k-1) + fk_quadro(x,k))/(2*h^2);
%             i_indices(cont) = k;
%             j_indices(cont) = k;
%             values(cont) = H_kk;
%             cont = cont +1;
%         else
%             H_kk = (fk_quadro(x+2*he_k, k-1) + fn_quadro(x+2*he_k) -  2*fk_quadro(x+he_k, k-1) -2*fn_quadro(x+he_k) + fk_quadro(x,k-1) + fn_quadro(x))/(2*h^2);
%             i_indices(cont) = k;
%             j_indices(cont) = k;
%             values(cont) = H_kk;
%             cont = cont +1;
%         end
% 
%         % Elementi fuori diagonale H(k, k+1)
%         he_k1 = zeros(n,1);
%         he_k1(k+1) = h;
%         if k < n
%             H_k_k1 = (fk_quadro(x+he_k1 +he_k,k) - fk_quadro(x+he_k, k) - fk_quadro(x+he_k,k) - fk_quadro(x, k))/(2*h^2); 
%             i_indices(cont) = k;
%             j_indices(cont) = k+1;
%             values(cont) = H_k_k1;
%             cont = cont +1;
% 
%             % impongo la simmetria
%             i_indices(cont) = k+1;
%             j_indices(cont) = k;
%             values(cont) = H_k_k1;
%             cont = cont+1;
%         else
%             % Caso circolare: H(n, 1)
%             he_n = zeros(n,1);
%             he_n(k) = h;
%             he_1 = zeros(n,1);
%             he_1(k) = h;
%             H_n1 = (fn_quadro(x+he_n +he_1) - fn_quadro(x+he_n) - fn_quadro(x+he_1) - fn_quadro(x))/(2*h^2);
%             i_indices(cont) = 1;
%             j_indices(cont) = n;
%             values(cont) = H_n1;
%             cont = cont +1;
% 
%             % impongo la simmetria
%             i_indices(cont) = n;
%             j_indices(cont) = 1;
%             values(cont) = H_n1;
%             cont = cont +1;
%         end
% 
%     end
% 
%     % Creazione della matrice Hessiana sparsa
%     hessian_approx = sparse(i_indices, j_indices, values, n, n);
% end
% 
% 
% tic;
% hessian = sparse_hessian(50*ones(1e4, 1), 1e-8);
% time = toc

h = 1e-12;
gradf_approx = @(x) findiff_grad_64(x,h);

gradf([2;5;9;-2;0])
gradf_approx([2;5;9;-2;0])
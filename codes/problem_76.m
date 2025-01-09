%% PROBLEMA 76
% ha minimo pari a 0 nell'origine
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


% implementing the function, the gradient and the hessiano for problem 76
function val = function_pb76(x)
    n = length(x);

    val = (x(n) - x(1)^2/10)^2;
    for k = 1:n-1
        val = val + (x(k) - x(k+1)^2/10)^2;
    end

    val = 0.5*val;

end

f = @(x) function_pb76(x);

function val = grad_pb76(x)
    n = length(x);

    val = zeros(n, 1);
    val(1,1) = (x(n) - x(1)^2/10) * (-0.2*x(1)) + (x(1) - x(2)^2/10);
    val(n, 1) = (x(n-1) - x(n)^2/10) *(-0.2*x(n)) + (x(n) - x(1)^2/10);

    for k =2:n-1
        val(k,1) = (x(k-1) - x(k)^2/10) * (-0.2*x(k)) + (x(k) - x(k+1)^2/10);
    end
end

gradf = @(x) grad_pb76(x);


function val = hessian_pb76(x)
    n = length(x);
    diags = zeros(n,5); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior

    % principal diag
    diags(2:n,1) = -0.2*x(1:n-1) + 3/50 *x(2:n).^2 +1;
    diags(1,1) =  -0.2*x(n) + 3/50 *x(1)^2 +1;

    % inferior diagonal
    diags(1:n-1,3) = -0.2*x(2:n);

    %superior diagonal
    diags(2:n,2) =  -0.2*x(2:n);

    % 2-inf e 2-suo diag
    diags(1, 5) = -x(1)/5;
    diags(n, 4) =-x(1)/5;

    val = spdiags(diags, [0,1,-1, n-1, - (n-1)], n,n);
end


Hessf = @(x) hessian_pb76(x);

tol = 1e-4;
iter_max = 300;


%% RUNNING THE EXPERIMENTS ON NEALDER MEAD
format short e

% setting the dimensionality
dimension = [10 25 50];


% initializing the structures to store some stats
execution_time_SX = zeros(length(dimension),11);
failure_struct_SX = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_SX = zeros(length(dimension),11);
fbest_struct_SX = zeros(length(dimension),11);
roc_struct_SX = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);

    % defining the given initial point
    x0 = 2*ones(n,1);

    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    rng(seed);
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
    disp(['**** SIMPLEX METHOD FOR THE PB 76 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

    disp(['Time: ', num2str(execution_time_SX(dim,1)), ' seconds']);

    disp('**** SIMPLES METHOD : RESULTS *****')
    disp('************************************')
    disp(['f(xk): ', num2str(fbest)])
    disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
    disp(['Rate of Convergence: ', num2str(roc_struct_SX(dim,1))])
    disp('************************************')

    if (failure)
        disp('FAIL')
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

        disp(['**** SIMPLEX METHOD FOR THE PB 76 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_SX(dim,i+1)), ' seconds']);
    
        disp('**** SIMPLES METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
        disp(['Rate of Convergence: ', num2str(roc_struct_SX(dim,1))])
        disp('************************************')
    
        if (failure)
            disp('FAIL')
        else
            disp('SUCCESS')
            disp('************************************')
        end
        disp(' ')

    end
end


varNames = ["avg fbest", "avg num of iters", "avg time of exec (sec)", "n failure", "avg roc"];
rowNames = string(dimension');
TSX = table( round(sum(fbest_struct_SX,2)/11, 4), round(sum(iter_struct_SX,2)/11, 4), round(sum(execution_time_SX,2)/11, 4), sum(failure_struct_SX,2), round(sum(roc_struct_SX,2)/11, 4) ,'VariableNames', varNames, 'RowNames', rowNames);
format short e
display(TSX)




%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD
format short e

iter_max = 5000;

% setting the values for the dimension
dimension = [1e3 1e4 1e5];

param = [0.4, 1e-4, 40; 0.3, 1e-4, 28; 0.4, 1e-3, 36];


% initializing structures to store some stats
execution_time_MN = zeros(length(dimension),11);
failure_struct_MN = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_MN = zeros(length(dimension),11);
fbest_struct_MN = zeros(length(dimension),11);
gradf_struct_MN = zeros(length(dimension),11);
roc_struct_MN = zeros(length(dimension),11);
ultima_direz_discesa = zeros(length(dimension), 11);

for dim = 1:length(dimension)
    n = dimension(dim);

    [rho, c1, btmax] = deal(param(dim, 1), param(dim, 2), param(dim, 3));


    %defining the given initial point
    x0 = 2*ones(n,1);
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    rng(seed);
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, pk_scalare_gradf] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
    execution_time_MN(dim,1) = toc(t1);
    fbest_struct_MN(dim,1) = fbest;
    iter_struct_MN(dim,1) = iter;
    gradf_struct_MN(dim,1) = gradfk_norm;
    roc_struct_MN(dim,1) = compute_roc(xseq);
    ultima_direz_discesa(dim,1) = pk_scalare_gradf;

    disp(['**** MODIFIED NEWTON METHOD FOR THE PB 76 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

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
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, pk_scalare_gradf] = modified_Newton(f,gradf, Hessf, x0_rndgenerated(:,i), iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
        execution_time_MN(dim,i+1) = toc(t1);
        fbest_struct_MN(dim,i+1) = fbest;
        iter_struct_MN(dim,i+1) = iter;
        failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
        gradf_struct_MN(dim,i+1) = gradfk_norm;
        roc_struct_MN(dim,i+1) = compute_roc(xseq);
        ultima_direz_discesa(dim,i+1) = pk_scalare_gradf;

        disp(['**** MODIFIED NEWTON METHOD FOR THE PB 76 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_MN(dim,i+1)), ' seconds']);
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

% plotto pk_scalar_gradfk
bar(ultima_direz_discesa')
ylabel('cos(angolo)')
title('Ultimo valore assunto da t(pk)*gradfk/(norm_pk * norm_gradfk)')
legend({'dim = 1e3', 'dim = 1e4', 'dim = 1e5'}, "Box", 'on', 'Location', 'best')



varNames = ["avg fbest", "avg gradf_norm","avg num of iters", "avg time of exec (sec)", "n failure", "avg roc"];
rowNames = string(dimension');
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 , sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
format short e
display(TMN)




%% FINITE DIFFERENCES
clc

function grad_approx = findiff_grad_76(x, h, type_h)
    %   - type_h: indica se la derivata è calcolata con h costante o h
    %   relativo
    n = length(x);

    if isempty(type_h)
        type_h = 'COST';
    end

    switch type_h
        case 'COST'
            passo1 = h;
            passok = h;
            passon = h;
        case 'REL'
            passo1 = h*abs(x(1));
            passon = h*abs(x(n));
    end

    grad_approx = zeros(n,1);
    grad_approx(1,1) =  passo1*(4*x(1)  - 2/5 * x(2)^2 + (8*x(1)*(x(1)^2+passo1^2))/100 - 4/5 * x(n)*x(1))/(4*passo1);
    
    for k = 2:n-1
        if strcmp('REL', type_h)
            passok = h*abs(x(k));
        end
        grad_approx(k,1) = passok*(4*x(k)  - 2/5 * x(k+1)^2 + (8*x(k)*(x(k)^2+passok^2))/100 - 4/5 * x(k-1)*x(k))/(4*passok);
    end
    grad_approx(n,1) = passon*(4*x(n)  - 2/5 * x(1)^2 + (8*x(n)*(x(n)^2+passon^2))/100 - 4/5 * x(n-1)*x(n))/(4*passon);
    
end


function hessian_approx = findiff_hess_76(x, h, type_h)
    % Calcola la matrice Hessiana sparsa per la funzione f(x)
    % Input:
    %   - x: vettore colonna (punto in cui calcolare l'Hessiana)
    %   - h: passo o vettore per la differenzaz rispetto ad una componente
    %   - type_h: indica se la derivata è calcolata con h costante o h
    %   relativo
    % Output:
    %   - H: matrice Hessiana sparsa

    n = length(x); % Dimensione del problema

    if isempty(type_h)
        type_h = 'COST';
    end
    
    % Preallocazione per la struttura sparsa
    i_indices = zeros(3*n,1);
    j_indices = zeros(3*n,1);
    values = zeros(3*n,1);
    cont = 1;

    % Loop su k (dalla definizione della funzione)
    for k = 1:n
        % Elementi diagonali H(k, k)
        if k == 1
            switch type_h
                case 'REL'
                    passok = h*abs(x(k));
                case 'COST'
                    passok = h;
            end
            % H_kk = (fn_quadro(x+2*he_k) + fk_quadro(x+2*he_k, 1) - 2*fn_quadro(x+he_k) -2*fk_quadro(x+he_k,1) + fn_quadro(x) + fk_quadro(x,1))/(2*h^2);
            H_kk = (2*passok^2 - 2/5 * x(n)*passok^2+ 0.12 * x(k)^2*passok^2+ 0.24 * x(k)*passok^3 +0.14 * passok^4)/(2*passok^2); 
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;
        elseif k < n
            switch type_h
                case 'REL'
                    passok = h*abs(x(k));
                case 'COST'
                    passok = h;
            end
            % H_kk = (fk_quadro(x+2*he_k, k-1) + fk_quadro(x+2*he_k, k) -  2*fk_quadro(x+he_k, k-1) -2*fk_quadro(x+he_k,k) + fk_quadro(x,k-1) + fk_quadro(x,k))/(2*h^2);
            H_kk = (2*passok^2 - 2/5 * x(k-1)*passok^2+ 0.12 * x(k)^2*passok^2+ 0.24 * x(k)*passok^3 +0.14 * passok^4)/(2*passok^2); 
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;
        else
            switch type_h
                case 'REL'
                    passok = h*abs(x(n));
                case 'COST'
                    passok = h;
            end
            % H_kk = (fk_quadro(x+2*he_k, k-1) + fn_quadro(x+2*he_k) -  2*fk_quadro(x+he_k, k-1) -2*fn_quadro(x+he_k) + fk_quadro(x,k-1) + fn_quadro(x))/(2*h^2);
            H_kk = (2*passok^2 - 2/5 * x(n-1)*passok^2+ 0.12 * x(k)^2*passok^2+ 0.24 * x(k)*passok^3 +0.14 * passok^4)/(2*passok^2); 
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;
        end
         
        % Elementi fuori diagonale H(k, k+1)
        if k < n
            % H_k_k1 = (fk_quadro(x+he_k1 +he_k,k) - fk_quadro(x+he_k, k) - fk_quadro(x+he_k1,k) - fk_quadro(x, k))/(2*h^2); 
            H_k_k1 = (-2/5 *passok^2*x(k+1) - 1/5 * passok^3)/(2*passok^2);
            i_indices(cont) = k;
            j_indices(cont) = k+1;
            values(cont) = H_k_k1;
            cont = cont +1;

            % impongo la simmetria
            i_indices(cont) = k+1;
            j_indices(cont) = k;
            values(cont) = H_k_k1;
            cont = cont+1;
        else
            % Caso circolare: H(n, 1)
            H_n1 = (-2/5 *passok^2*x(1) - 1/5 * passok^3)/(2*passok^2);
            i_indices(cont) = 1;
            j_indices(cont) = n;
            values(cont) = H_n1;
            cont = cont +1;
        
            % impongo la simmetria
            i_indices(cont) = n;
            j_indices(cont) = 1;
            values(cont) = H_n1;
            cont = cont +1;
        end
        
    end

    % Creazione della matrice Hessiana sparsa
    hessian_approx = sparse(i_indices, j_indices, values, n, n);
end

h = 1e-10;
type_h = 'REL';
gradf_approx = @(x) findiff_grad_76(x,h, type_h);
Hessf_approx = @(x) findiff_hess_76(x,h, type_h);

vec = 0.5*ones(7,1);
vec = [0.2; 0.4; -0.2; 0.5; 0.1; -1; 0.1];

gradf(vec)
gradf_approx(vec)

full(Hessf(vec))
tic
full(Hessf_approx(vec))
time = toc


%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD WITH FIN DIFF
format short e
clc

iter_max = 5000;
tol = 1e-4;



% setting the values for the dimension
h_values = [1e-2 1e-4 1e-6 1e-8 1e-10 1e-12];
dimension = [1e3 1e4 1e5];
param = [0.4, 1e-4, 40; 0.3, 1e-4, 28; 0.4, 1e-3, 36];
type_h = 'COST';

% initializing structures to store some stats
execution_time_MN_h = zeros(length(dimension),6);
failure_struct_MN_h = zeros(length(dimension),6); 
iter_struct_MN_h = zeros(length(dimension),6);
fbest_struct_MN_h = zeros(length(dimension),6);
gradf_struct_MN_h = zeros(length(dimension),6);
roc_struct_MN_h = zeros(length(dimension),6);


for id_h = 1:length(h_values)

   h = h_values(id_h);

    gradf_approx = @(x) findiff_grad_76(x,h, type_h);
    hessf_approx = @(x) findiff_hess_76(x,h, type_h);

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
        x0 = 2*ones(n,1);
        
        % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
        rng(seed);
        x0_rndgenerated = zeros(n,10);
        x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
        
    
        % SOLVING MODIFIED NEWTON METHOD METHOD
        % first initial point
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf_approx, hessf_approx, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
        execution_time_MN(dim,1) = toc(t1);
        fbest_struct_MN(dim,1) = fbest;
        iter_struct_MN(dim,1) = iter;
        gradf_struct_MN(dim,1) = gradfk_norm;
        roc_struct_MN(dim,1) = compute_roc(xseq);
        disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 76 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);
    
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
            [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf_approx, hessf_approx, x0_rndgenerated(:,i), iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
            execution_time_MN(dim,i+1) = toc(t1);
            fbest_struct_MN(dim,i+1) = fbest;
            iter_struct_MN(dim,i+1) = iter;
            failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
            gradf_struct_MN(dim,i+1) = gradfk_norm;
            roc_struct_MN(dim,i+1) = compute_roc(xseq);
    
            disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 76 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);
    
            disp(['Time: ', num2str(execution_time_MN(dim,i+1)), ' seconds']);
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
    format short e
    display(TMN)

    execution_time_MN_h(:,id_h) = sum(execution_time_MN,2)/11;
    failure_struct_MN_h(:,id_h) = sum(failure_struct_MN,2); 
    iter_struct_MN_h(:,id_h) = sum(iter_struct_MN,2)/11;
    fbest_struct_MN_h(:,id_h) = sum(fbest_struct_MN,2)/11;
    gradf_struct_MN_h(:,id_h) = sum(gradf_struct_MN,2)/11;
    roc_struct_MN_h(:,id_h) = sum(roc_struct_MN,2)/11;

end


%% PROBLEMA 25
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
function val = function_pb25(x)
    n = length(x);
    val = 0;
    
    for k= 1:2:n-1
        val = val + 10*(x(k)^2 - x(k+1))^2;
    end
    for k=2:2:n-1
        val = val + (x(k-1) -1)^2;
    end

    if mod(n,2) == 1
        val = val + (10*x(n)^2 - x(1))^2;
    else
        val = val + (x(n-1) -1)^2;
    end

    val = 0.5*val;
end

f = @(x) function_pb25(x);

function grad = grad_pb25(x)
    n = length(x);
    grad = zeros(n,1);
    
    if mod(n,2) == 0
        grad(1:2:n-1) = 200*x(1:2:n-1).^3 - 200*x(1:2:n-1).*x(2:2:n) + x(1:2:n-1) -1;
        grad(2:2:n) = - 100*(x(1:2:n-1).^2 - x(2:2:n));
    else
        grad(1, 1) = 200*x(1)^3 - 200*x(1)*x(2) + x(1) -1 - 100*(x(n)^2 - x(1));
        grad(3:2:n-1) = 200*x(3:2:n-1).^3 - 200*x(3:2:n-1).*x(4:2:n) + x(3:2:n-1) -1;
        grad(2:2:n-2) = - 100*(x(1:2:n-3).^2 - x(2:2:n-2));
        grad(n,1) = 200*x(n).^3  - 200*x(n)*x(1) + x(n) -1;
    end
end

gradf = @(x) grad_pb25(x);


function val = hessian_pb25(x)
    n = length(x);
    diags = zeros(n,5); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior

    % principal diag
    if mod(n,2) == 0
        diags(2:2:n,1) = 100;
        diags(1:2:n-1,1) = 600*x(1:2:n-1).^2 - 200 * x(2:2:n) +1;
    else
        diags(1,1) = 600*x(1)^2  - 200*x(2) +101;
        diags(2:2:n,1) = 100;
        diags(3:2:n-1,1) = 600*x(3:2:n-1).^2 - 200 * x(4:2:n) +1;
        diags(n,1) = 600*x(n).^2 - 200*x(1) +1;
    end

    % inferior diagonal
    diags(1:2:n-1,3) = -200*x(1:2:n-1);
    diags(2:2:n-2, 3) = 0;

    %superior diagonal
    diags(3:2:n-1,2) = 0;
    diags(2:2:n, 2) = -200*x(1:2:n-1);

    % these diagonals exists only if n is odd
    if mod(n,2) == 1
        diags(1,5) = - 200*x(n);
        diags(n,4) = - 200*x(n);
    end


    val = spdiags(diags, [0,1,-1, n-1, - (n-1)], n,n);
end


Hessf = @(x) hessian_pb25(x);

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
    x0(1:2:n) = -1.2;

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
    disp(['**** SIMPLEX METHOD FOR THE PB 25 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

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
        [~,xseq,iter,fbest, ~, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        execution_time_SX(dim,i+1) = toc(t1);
        fbest_struct_SX(dim,i+1) = fbest;
        iter_struct_SX(dim,i+1) = iter;
        failure_struct_SX(dim,i+1) = failure_struct_SX(dim,i+1) + failure;
        roc_struct_SX(dim,i+1) = compute_roc(xseq);

        disp(['**** SIMPLEX METHOD FOR THE PB 25 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_SX(dim,i+1)), ' seconds']);
    
        disp('**** SIMPLES METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
        disp(['Rate of Convergence: ', num2str(roc_struct_SX(dim,i+1))])
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
display(TSX)



%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD
format short e
clc

% setting the values for the dimension
dimension = [1e3 1e4 1e5];
iter_max = 3000;

param = [0.5, 1e-4, 48; 0.5, 1e-4, 48; 0.5, 1e-4, 48];

rng(seed);

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
    x0 = ones(n,1);
    x0(1:2:n) = -1.2;
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
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
    disp(['**** MODIFIED NEWTON METHOD FOR THE PB 25 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

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

        disp(['**** MODIFIED NEWTON METHOD FOR THE PB 25 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_MN(dim,i+1)), ' seconds']);
        disp(['Backtracking parameters (rho, c1): ', num2str(rho), ' ', num2str(c1)]);
    
        disp('**** MODIFIED NEWTON METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['norma di gradf(xk): ', num2str(gradfk_norm)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max)])
        disp(['Rate of Convergence: ', num2str(roc_struct_MN(dim,i+1))])
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
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
display(TMN)




%% FINITE DIFFERENCES
clc

function grad_approx = findiff_grad_25(x, h, type_h)
    n = length(x);
    grad_approx = zeros(n,1);
    


    if mod(n,2) == 0
        % CASO n PARI
        for k = 1:n
            switch type_h
                case 'REL'
                    passok = h * abs(x(k));
                case 'COST'
                    passok = h;
            end
    
            if mod(k,2) == 0
                grad_approx(k,1) = (-40*passok*(10*x(k-1)^2 - 10*x(k)))/(4*passok);
            else
                grad_approx(k,1) = (80 * x(k)*passok*(10*x(k)^2 + 10*passok^2 - 10*x(k+1)) + 4*passok*(x(k)-1))/(4*passok);
            end
        end
    else
        % CASO n DISPARI
        for k = 1:n
            switch type_h
                case 'REL'
                    passok = h * abs(x(k));
                case 'COST'
                    passok = h;
            end
    
            if k == 1
                grad_approx(1,1) = (80 * x(k)*passok*(10*x(k)^2 + 10*passok^2 - 10*x(k+1)) + 4*passok*(x(k)-1) - 40*passok*(10*x(n)^2 - 10*x(1)))/(4*passok);
            elseif k == n
                 grad_approx(k,1) = (80 * x(k)*passok*(10*x(k)^2 + 10*passok^2 -10*x(1)))/(4*passok);
            elseif mod(k,2) == 0
                grad_approx(k,1) = (-40*passok*(10*x(k-1)^2 - 10*x(k)))/(4*passok);
            else
                grad_approx(k,1) = (80 * x(k)*passok*(10*x(k)^2 + 10*passok^2 - 10*x(k+1)) + 4*passok*(x(k)-1))/(4*passok);
            end
        end
    end
end


function hessian_approx = findiff_hess_25(x, h, type_h)
    % Calcola la matrice Hessiana sparsa per la funzione f(x)
    % Input:
    %   - x: vettore colonna (punto in cui calcolare l'Hessiana)
    %   - h: passo
    % Output:
    %   - H: matrice Hessiana sparsa

    n = length(x); % Dimensione del problema
    
    % Preallocazione per la struttura sparsa
    i_indices = [];
    j_indices = [];
    values = [];
    cont = 1;

    % Loop su k (dalla definizione della funzione)
    for k = 1:n

        % ELEMENTI DIAGONALI
        if k == 1 && mod(n,2) == 1
            % caso k == 1 con n dispari

            switch type_h 
                case 'COST'
                    hk = h;
                case 'REL'
                    hk = h*abs(x(k));
            end

            H_kk = (40*hk^2*(10*x(k)^2 -10*x(k+1)) + 1400*hk^4 + 2400*hk^3*x(k) + 800*x(k)^2*hk^2 + 2*hk^2 + 200*hk^2)/(2*hk^2);
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;

        
        elseif k == n && mod(n,2) == 1
            % caso k pari con n pari

            switch type_h 
                case 'COST'
                    hk = h;
                case 'REL'
                    hk = h*abs(x(k));
            end

            H_kk = (40*hk^2*(10*x(k)^2 -10*x(1)) + 1400*hk^4 + 2400*hk^3*x(k) + 800*x(k)^2*hk^2)/(2*hk^2);
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;


        elseif mod(k,2) == 1
            % caso k dispari (se n dispari, non entra qui ma nella condizione sopra)
            switch type_h 
                case 'COST'
                    hk = h;
                case 'REL'
                    hk = h*abs(x(k));
            end

            H_kk = (40*hk^2*(10*x(k)^2 -10*x(k+1)) + 1400*hk^4 + 2400*hk^3*x(k) + 800*x(k)^2*hk^2 + 2*hk^2)/(2*hk^2);
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;

        elseif mod(k,2) == 0 
            % caso k pari con n pari
            switch type_h 
                case 'COST'
                    hk = h;
                case 'REL'
                    hk = h*abs(x(k));
            end

            H_kk = (200*hk^2)/(2*hk^2);
            i_indices(cont) = k;
            j_indices(cont) = k;
            values(cont) = H_kk;
            cont = cont +1;

        end

        % ELEMENTI EXTRA DIAG
        if mod(n,2) == 1 && k == n
            % ho le due diagonali estremali
            
            switch type_h
                case 'COST'
                    h1 = h;
                case 'REL'
                    h1 =h*abs(x(1));
            end

            H_n1 = (20*h1 *(-20*x(k)* hk - 10*hk^2))/(2*hk*h1);
            i_indices(cont) = n;
            j_indices(cont) = 1;
            values(cont) = H_n1;
            cont = cont +1;

            % impongo la simmetria
            i_indices(cont) = 1;
            j_indices(cont) = n;
            values(cont) = H_n1;
            cont = cont +1;

        elseif mod(k,2) == 1 && k < n
            % ho solo le derivate k, k+1 con k dispari
            
            switch type_h
                case 'COST'
                    hk1 = h;
                case 'REL'
                    hk1 =h*abs(x(k+1));
            end

            H_k_k1 = (20*hk1*(-10 * hk^2 - 20*hk*x(k)))/(2*hk*hk1);
            i_indices(cont) = k;
            j_indices(cont) = k+1;
            values(cont) = H_k_k1;
            cont = cont +1;

            % impongo la simmetria
            i_indices(cont) = k+1;
            j_indices(cont) = k;
            values(cont) = H_k_k1;
            cont = cont +1;
        end

    end

   
    % Creazione della matrice Hessiana sparsa
    hessian_approx = sparse(i_indices, j_indices, values, n, n);
end


h = 1e-2;
type_h = 'COST';
gradf_approx = @(x) findiff_grad_25(x,h, type_h);
Hessf_approx = @(x) findiff_hess_25(x,h, type_h);

vec = [1; 0.5*ones(13,1); 1];
% vec = [0.2; 0.4; -0.2; 0.5; 0; 0.3; 0];

gradf(vec)
gradf_approx(vec)

format short
full(Hessf(vec))
tic
full(Hessf_approx(vec))
time = toc

%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD WITH FIN DIFF
format short e
clc

iter_max = 3000;
tol = 1e-3;



% setting the values for the dimension
h_values = [1e-2 1e-4 1e-6 1e-8 1e-10 1e-12];
dimension = [1e3 1e4 1e5];
param = [0.5, 1e-4, 48; 0.5, 1e-4, 48; 0.5, 1e-4, 48;];
type_h = 'REL';

tables = struct;

% initializing structures to store some stats
execution_time_MN_h = zeros(length(dimension),6);
failure_struct_MN_h = zeros(length(dimension),6); 
iter_struct_MN_h = zeros(length(dimension),6);
fbest_struct_MN_h = zeros(length(dimension),6);
gradf_struct_MN_h = zeros(length(dimension),6);
roc_struct_MN_h = zeros(length(dimension),6);


for id_h = 1:length(h_values)

   h = h_values(id_h);
   tol = min(1e-3, h);

    gradf_approx = @(x) findiff_grad_25(x,h, type_h);
    hessf_approx = @(x) findiff_hess_25(x,h, type_h);

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
        x0(1:2:n) = -1.2;
        
        % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
        rng(seed);
        x0_rndgenerated = zeros(n,10);
        x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
        
    
        % SOLVING MODIFIED NEWTON METHOD METHOD
        % first initial point
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, ~] = modified_Newton(f,gradf_approx, hessf_approx, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
        execution_time_MN(dim,1) = toc(t1);
        fbest_struct_MN(dim,1) = fbest;
        iter_struct_MN(dim,1) = iter;
        gradf_struct_MN(dim,1) = gradfk_norm;
        roc_struct_MN(dim,1) = compute_roc(xseq);
        disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 25 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);
    
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
            [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, ~] = modified_Newton(f,gradf_approx, hessf_approx, x0_rndgenerated(:,i), iter_max, rho, c1, btmax, tol, [], 'ALG', 0);       
            execution_time_MN(dim,i+1) = toc(t1);
            fbest_struct_MN(dim,i+1) = fbest;
            iter_struct_MN(dim,i+1) = iter;
            failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
            gradf_struct_MN(dim,i+1) = gradfk_norm;
            roc_struct_MN(dim,i+1) = compute_roc(xseq);
    
            disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 25 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);
    
            disp(['Time: ', num2str(execution_time_MN(dim,i+1)), ' seconds']);
            disp(['Backtracking parameters (rho, c1): ', num2str(rho), ' ', num2str(c1)]);
        
            disp('**** MODIFIED NEWTON METHOD : RESULTS *****')
            disp('************************************')
            disp(['f(xk): ', num2str(fbest)])
            disp(['norma di gradf(xk): ', num2str(gradfk_norm)])
            disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max)])
            disp(['Rate of Convergence: ', num2str(roc_struct_MN(dim,i+1))])
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

    tables.(['Table' num2str(id_h)]) = TMN;

    execution_time_MN_h(:,id_h) = sum(execution_time_MN,2)/11;
    failure_struct_MN_h(:,id_h) = sum(failure_struct_MN,2); 
    iter_struct_MN_h(:,id_h) = sum(iter_struct_MN,2)/11;
    fbest_struct_MN_h(:,id_h) = sum(fbest_struct_MN,2)/11;
    gradf_struct_MN_h(:,id_h) = sum(gradf_struct_MN,2)/11;
    roc_struct_MN_h(:,id_h) = sum(roc_struct_MN,2)/11;

end


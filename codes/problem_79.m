% PROBLEMA 79

close all
clear all
clc

% setting the seed
seed = min(339268, 343310); 

% function to compute the rate of convergence
function rate_of_convergence = compute_roc(x_esatto, xseq)
if size(xseq,2) >=3
    rate_of_convergence = log(norm(x_esatto - xseq(:,end))/norm(x_esatto - xseq(:, end-1)))/log(norm(x_esatto - xseq(:,end-1))/norm(x_esatto - xseq(:, end-2)));
else 
    rate_of_convergence = nan;
end
end


% implementing the function, the gradient and the hessiano for problem 79
function val = function_pb79(x)
    % definisco preliminarm i fk per comodità
    x_esteso = [0; x; 0];
    n = length(x_esteso);
    fk = @(x,k) (3 + x(k)/10)*x(k) +1 - x(k-1) - 2*x(k+1);

    val = 0;
    for k = 2:n-1
        val = val + fk(x_esteso, k);
    end

    val = 0.5*val;

end

f = @(x) function_pb79(x);

function val = grad_pb79(x)
    n = length(x) + 2;
    fk = @(x,k) (3 + x(k)/10)*x(k) +1 - x(k-1) - 2*x(k+1);
    x_esteso = [0; x; 0];

    val = zeros(n, 1);
    val(1,1) = fk(x_esteso, 2)*(3-0.2*x(1)) - fk(x_esteso, 3);
    val(n,1) = -2 * fk(x_esteso, n-3) + fk(x_esteso, n-2);

    for k =3:n-2
        val(k,1) = -2 * fk(x_esteso, k-1) + fk(x_esteso, k) - fk(x_esteso, k+1);
    end
end

gradf = @(x) grad_pb79(x);


function val = hessian_pb79(x)
    n = length(x);
    diags = zeros(n,3); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior

    % principal diag
    diags(2:n-1,1) = 5 + (3-0.2 * x(2:n-1)).^2;
    diags(1,1) = 1 + (3-0.2 * x(1))^2;
    diags(n,1) = 4 + (3-0.2 * x(n))^2;

    % inferior diagonal
    diags(1:n-1,3) = -2*(3-0.2*x(1:n-1)) - (3-0.2*x(2:n));

    %superior diagonal
    diags(2:n,2) =  -2*(3-0.2*x(1:n-1)) - (3-0.2*x(2:n));

    val = spdiags(diags, [0,1,-1], n,n);
end


Hessf = @(x) hessian_pb79(x);

n = 1e3;

% minimum point is such that gradf(x) = 0
x_esatto = fsolve(gradf, -0.7*ones(n,1));
f(x_esatto)


%% RUNNING THE EXPERIMENTS ON NEALDER MEAD
format short e

% setting the dimensionality
dimension = [10 25 50];
rng(seed);


% initializing the structures to store some stats
execution_time_SX = zeros(length(dimension),11);
failure_struct_SX = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_SX = zeros(length(dimension),11);
fbest_struct_SX = zeros(length(dimension),11);
roc_struct_SX = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);
    x_esatto = -2*1e-5 * ones(n,1);

    % defining the given initial point
    x0 = (1:1:n)';

    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);

    % SOLVING SIMPLEX METHOD
    % first initial point
    fprintf('solving the SX method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [~, xseq,iter,fbest, ~, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    execution_time_SX(dim,1) = toc(t1);
    fbest_struct_SX(dim,1) = fbest;
    iter_struct_SX(dim,1) = iter;
    roc_struct_SX(dim,1) = compute_roc(x_esatto, xseq);

    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_SX(dim,1) = failure_struct_SX(dim,1) + failure;

    for i = 1:10
        fprintf('solving the SX method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [~,~,iter,fbest, ~, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        execution_time_SX(dim,i+1) = toc(t1);
        fbest_struct_SX(dim,i+1) = fbest;
        iter_struct_SX(dim,i+1) = iter;
        failure_struct_SX(dim,i+1) = failure_struct_SX(dim,i+1) + failure;
        roc_struct_SX(dim,i+1) = compute_roc(x_esatto, xseq);
    end
end


varNames = ["average fbest", "average number of iterations", "average time of execution (sec)", "numbers of failure", "average rate of convergence"];
rowNames = string(dimension');
TSX = table(sum(fbest_struct_SX,2)/11, sum(iter_struct_SX,2)/11, sum(execution_time_SX,2)/11, sum(failure_struct_SX,2),sum(roc_struct_SX,2)/11 ,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TSX)



%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD
format short e

iter_max = 6000;

% setting the values for the dimension
dimension = [1e3 1e4 1e5];
% VALORI USATI PER 1e4 e 1e5
% rho = 0.7; c1 = 1e-3; btmax = 98; tau_kmax = 1e4; tol = 1e-4;
% VALORI PER 1e3
rho = 0.9;  c1 = 1e-6; btmax = 150; tau_kmax = 1e4; tol = 1e-4;

rng(seed);

% % initializing structures to store some stats
% execution_time_MN = zeros(length(dimension),11);
% failure_struct_MN = zeros(length(dimension),11); %for each dimension we count the number of failure
% iter_struct_MN = zeros(length(dimension),11);
% fbest_struct_MN = zeros(length(dimension),11);
% gradf_struct_MN = zeros(length(dimension),11);
% roc_struct_MN = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);
    x_esatto = -2*1e-5 * ones(n,1);


    %defining the given initial point
    x0 = (1:1:n)';
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    fprintf('solving the MN method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax, x_esatto);       
    execution_time_MN(dim,1) = toc(t1);
    fbest_struct_MN(dim,1) = fbest;
    iter_struct_MN(dim,1) = iter;
    gradf_struct_MN(dim,1) = gradfk_norm;
    roc_struct_MN(dim,1) = compute_roc(x_esatto, xseq);

    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_MN(dim,1) = failure_struct_MN(dim,1) + failure ;

    for i = 1:10
        fprintf('solving the MN method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax, x_esatto);       
        execution_time_MN(dim,i+1) = toc(t1);
        fbest_struct_MN(dim,i+1) = fbest;
        iter_struct_MN(dim,i+1) = iter;
        failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
        gradf_struct_MN(dim,i+1) = gradfk_norm;
        roc_struct_MN(dim,i+1) = compute_roc(x_esatto, xseq);
    end
end


varNames = ["average fbest", "average gradf_norm","average number of iterations", "average time of execution (sec)", "numbers of failure", "average rate of convergence"];
rowNames = string(dimension');
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TMN)


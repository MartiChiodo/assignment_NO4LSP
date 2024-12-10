%% ESERCIZIO 3 (marti)

clear all
clc

% setting the seed
seed = min(339268, 343310); %poi andrà modificato opportunamente

% Let's begin by implementing the PENALTY FUNCTION 1
% The function is F : R^n --> R (scalar function)

function Fx = PF1_funct(x)
    % x is a matrix, each col contains a vector of dimension n
    % Fx is a vector, the i-th element is F(x(:,i))

    Fx = zeros(1,size(x,2));
    for col = 1:size(x,2)
        Fx(1,col) = 0.5* 1e-5 * sum((x(:,col) - ones(size(x,1),1)).^2) + 0.5*(sum(x(:,col).^2) - 0.25)^2;
    end
end

f = @(x) PF1_funct(x);
gradf = @(x) 1e-5.*(x-ones(length(x),1)) + 2*(sum(x.^2) -0.25).*x;

function hessf = hessian(x)
    n = length(x);
    diags = zeros(n,3); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior
    diags(1:n,1) = 1e-5 + 4*x(1:n).^2 + 2*(sum(x(:,1).^2) -0.25);
    % inferior diagonal
    diags(1:n-1,3) = 4.*x(1:n-1).*x(2:n);
    %superior diagonal
    diags(2:n,2) = 4.*x(2:n).*x(1:n-1);
    hessf = spdiags(diags, [0,1,-1], n,n);
end

Hessf = @(x) hessian(x);

% common parameters for the methods
iter_max = 200;
tol = 1e-7;


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

for dim = 1:length(dimension)
    n = dimension(dim);

    % defining the given initial point
    x0 = (1:1:n)';

    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);

    % SOLVING SIMPLEX METHOD
    % first initial point
    fprintf('solving the SX method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [~, ~,iter,fbest, ~, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    execution_time_SX(dim,1) = toc(t1);
    fbest_struct_SX(dim,1) = fbest;
    iter_struct_SX(dim,1) = iter;

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
    end
end


varNames = ["average fbest", "average number of iterations", "average time of execution (sec)", "numbers of failure"];
rowNames = string(dimension');
TSX = table(sum(fbest_struct_SX,2)/11, sum(iter_struct_SX,2)/11, sum(execution_time_SX,2)/11, sum(failure_struct_SX,2), 'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TSX)



%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD
format short e

iter_max = 5000;

% setting the values for the dimension
dimension = [1e3 1e4 1e5];
rho = 0.5; c1 = 1e-4; btmax = 45; tau_kmax = 1e4; tol = 1e-5;
rng(seed);

% initializing structures to store some stats
execution_time_MN = zeros(length(dimension),11);
failure_struct_MN = zeros(length(dimension),11); %for each dimension we count the number of failure
iter_struct_MN = zeros(length(dimension),11);
fbest_struct_MN = zeros(length(dimension),11);
gradf_struct_MN = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);

    %defining the given initial point
    x0 = (1:1:n)';
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    fprintf('solving the MN method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax);       
    execution_time_MN(dim,1) = toc(t1);
    fbest_struct_MN(dim,1) = fbest;
    iter_struct_MN(dim,1) = iter;
    gradf_struct_MN(dim,1) = gradfk_norm;

    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_MN(dim,1) = failure_struct_MN(dim,1) + failure ;

    for i = 1:10
        fprintf('solving the MN method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax);       
        execution_time_MN(dim,i+1) = toc(t1);
        fbest_struct_MN(dim,i+1) = fbest;
        iter_struct_MN(dim,i+1) = iter;
        failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
        gradf_struct_MN(dim,i+1) = gradfk_norm;
    end
end


varNames = ["average fbest", "average gradf_norm","average number of iterations", "average time of execution (sec)", "numbers of failure"];
rowNames = string(dimension');
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2),'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TMN)



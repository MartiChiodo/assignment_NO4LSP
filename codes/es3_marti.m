%% ESERCIZIO 3 (marti)

clear all
clc

% setting the seed
seed = min(339268, 343310); %poi andrà modificato opportunamente
rng(seed);

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
gradf = @(x) x .* (2+1e-5) -1e-5;
Hessf = @(x) spdiags((2+1e-5)*ones(length(x),1), 0, length(x), length(x));


% setting the values for the dimension
dimension = [1e2 1e3];
iter_max = 100;
tol = 1e-7;
avg_execution_time_SX = zeros(length(dimension),1);
failure_struct_SX = zeros(length(dimension),1); %for each dimension we count the number of failure
avg_execution_time_MN = zeros(length(dimension),1);
failure_struct_MN = zeros(length(dimension),1); %for each dimension we count the number of failure


 % initializing structures
iter_struct_SX = zeros(length(dimension),1);
fbest_struct_SX = zeros(length(dimension),1);
iter_struct_MN = zeros(length(dimension),1);
fbest_struct_MN = zeros(length(dimension),1);

for dim = 1:length(dimension)
    n = dimension(dim);
    x0 = (1:1:n)';
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    %
    f_best_avg_SX = 0;
    iter_avg_SX = 0;
    f_best_avg_MN = 0;
    iter_avg_MN = 0;

    % I can measure the computanional averge time
    time_SX = 0;
    time_MN = 0;

    % SOLVING SIMPLEX METHOD
    % first initial point
    fprintf('solving the SX method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [~, ~,iter,fbest, ~, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    time_SX = time_SX + toc(t1);
    f_best_avg_SX = f_best_avg_SX + fbest;
    iter_avg_SX = iter_avg_SX + iter;


    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_SX(dim,1) = failure_struct_SX(dim,1) + failure;

    for i = 1:10
        fprintf('solving the SX method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [~,~,iter,fbest, ~, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        time_SX = time_SX + toc(t1);
        f_best_avg_SX = f_best_avg_SX + fbest;
        iter_avg_SX = iter_avg_SX + iter;
        failure_struct_SX(dim,1) = failure_struct_SX(dim,1) + failure;
    end

    fbest_struct_SX(dim,1) = f_best_avg_SX/11;
    iter_struct_SX(dim,1) = iter_avg_SX/11;

    % I store the average execution time
    avg_execution_time_SX(dim,1) = time_SX/11;


    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    rho = 0.5; c1 = 1e-4; btmax = 50; tau_kmax = 1e4; tol = 1e-7;
    fprintf('solving the MN method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax);       
    time_MN = time_MN + toc(t1);
    f_best_avg_MN = f_best_avg_MN + fbest;
    iter_avg_MN = iter_avg_MN + iter;


    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_MN(dim,1) = failure_struct_MN(dim,1) + failure;

    for i = 1:10
        fprintf('solving the MN method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] = modified_Newton(f,gradf, Hessf, x0, iter_max, rho, c1, btmax, tol, tau_kmax);       
        time_MN = time_MN + toc(t1);
        f_best_avg_MN = f_best_avg_MN + fbest;
        iter_avg_MN = iter_avg_MN + iter;
        failure_struct_MN(dim,1) = failure_struct_MN(dim,1) + failure;
    end

    fbest_struct_MN(dim,1) = f_best_avg_MN/11;
    iter_struct_MN(dim,1) = iter_avg_MN/11;

    % I store the average execution time
    avg_execution_time_MN(dim,1) = time_MN/11;

end


TSX = table( dimension', fbest_struct_SX, iter_struct_SX, avg_execution_time_SX, failure_struct_SX);
format bank
display(TSX)

TMN = table( dimension', fbest_struct_MN, iter_struct_MN, avg_execution_time_MN, failure_struct_MN);
format bank
display(TMN)



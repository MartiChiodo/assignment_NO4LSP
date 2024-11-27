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
Hessf = @(x) (2+1e-5)*eye(length(x));


% setting the values for the dimension
dimension = [1e2 1e3 1e4];
iter_max = 100;
tol = 1e-9;
avg_execution_time_SX = zeros(3,1);
failure_struct_SX = zeros(3,1); %for each dimension we count the number of failure

 % initializing structures
iter_struct_SX = zeros(1,3);
fbest_struct_SX = zeros(1,3);


for dim = 1:3
    n = dimension(dim);
    x0 = (1:1:n)';
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    
    % conditions for backtracking (poi le sistemo quando implementiamo il secondo metodo)
    ro = 0.5;
    c = 1e-4;

    %
    f_best_avg = 0;
    x_best_avg = zeros(n,1);
    iter_avg = 0;

    % I can measure the computanional averge time
    time = 0;

    % first initial point
    fprintf('solving the NM method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [xbest, ~,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    time = time + toc(t1);
    f_best_avg = f_best_avg + fbest;
    iter_avg = iter_avg + iter;


    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_SX(dim) = failure_struct_SX(dim) + failure;

    for i = 1:10
        fprintf('solving the NM method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [xbest,~,iter,fbest, flag, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        time = time + toc(t1);
        f_best_avg = f_best_avg + fbest;
        x_best_avg = x_best_avg + xbest;
        iter_avg = iter_avg + iter;

        % if failure = true (failure == 1), the run was unsuccessful; otherwise
        % failure = 0
        failure_struct_SX(dim) = failure_struct_SX(dim) + failure;
    end

    fbest_struct_SX(dim) = f_best_avg/11;
    iter_struct_SX(1,dim) = iter_avg/11;

    % I store the average execution time
    avg_execution_time_SX(dim,1) = time/11;

end

failure_struct_SX
avg_execution_time_SX
fbest_struct_SX
iter_struct_SX


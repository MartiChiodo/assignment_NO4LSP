%% ESERCIZIO 3 (marti)
clear all
clc

% setting the seed
seed = 1234; %poi andrà modificato opportunamente
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
dimension = [1e1 1e2 1e3];
iter_max = 100;
tol = 1e-9;
avg_execution_time_NM = zeros(3,1);
failure_struct_NM = zeros(3,1); %for each dimension we count the number of failure

for dim = 1:3
    n = dimension(dim);
    x0 = (1:1:n)';
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    
    % conditions for backtracking (poi le sistemo quando implementiamo il secondo metodo)
    ro = 0.5;
    c = 1e-4;

    % initializing structures
    xbest_struct = zeros(n, 11);
    iter_struct = zeros(1,11);
    fbest_struct = zeros(1,11);

    % I can measure the computanional averge time
    time = 0;

    % first initial point
    fprintf('solving the NM method for the first x0 with dim = %i \n', n)
    t1 = tic;
    [xbest, ~,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],iter_max*size(x0,1),tol);
    time = time + toc(t1);
    xbest_struct(:,1) = xbest;
    iter_struct(1,1) = iter;
    fbest_struct(1,1) = fbest;

    % if failure = true (failure == 1), the run was unsuccessful; otherwise
    % failure = 0
    failure_struct_NM(dim) = failure_struct_NM(dim) + failure;

    for i = 1:10
        fprintf('solving the NM method for the %i -th x0 with dim = %i \n', i+1, n)
        t1 = tic;
        [xbest,~,iter,fbest, flag, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max*size(x0,1),tol);
        time = time + toc(t1);
        xbest_struct(:,i+1) = xbest;
        iter_struct(1,i+1) = iter;
        fbest_struct(1,i+1) = fbest;

        % if failure = true (failure == 1), the run was unsuccessful; otherwise
        % failure = 0
        failure_struct_NM(dim) = failure_struct_NM(dim) + failure;
    end


    fbest_struct
    iter_struct

    % I store the average execution time
    avg_execution_time_NM(dim,1) = time/11;

end

failure_struct_NM
avg_execution_time_NM


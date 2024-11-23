%% ESERCIZIO 3 (marti)
clear all
clc

% setting the seed
seed = 1234; %poi andrà modificato opportunamente
rng(seed);

% Let's begin by implementing the BANDED TRIGONOMETRIC PB
% The function is F : R^n --> R (scalar function)

function Fx = BTP_func(x)
    % x is a column vector of dimension n
    % Fx is a scalar 
    Fx = 0;
    n = length(x);
    Fx = Fx + 1 - cos(x(1)) -sin(x(2)) + n*(1-cos(x(n)) + sin(x(n-1))); %step for i = 1 and i = n
    for i = 2:n-1
        Fx = Fx + i*(1 - cos(x(i)) + sin(x(i-1)) - sin(x(i+1)));
    end
end

function grad_x = BTP_grad(x)
    % x is a column vector of dimension n
    % grad_x is a column vector of dimension n
    n = length(x);
    grad_x = zeros(n,1);
    grad_x(1) = sin(x(1)) + 2*cos(x(1));
    grad_x(n) = (n-1) * (-cos(x(n))) + n*sin(x(n));
    grad_x(2:n-1) = (2:n-1) * sin(x(2:n-1)) + (3:n)*cos(x(2:n-1)) - (1:n-2)* cos(x(2:n-1));
end

function Hess_x = BTP_Hess(x)
    % x is a column vector of dimension n
    % Hess_x is a matrix n x n

    % Notice that all the entries of the Hessian are zeros except for the
    % diagonal
    n = length(x);
    diagonale = zeros(n,1);
    diagonale(1) = cos(x(1)) - 2*sin(x(1));
    diagonale(n) = (n-1) * sin(x(n)) + n* cos(x(n));
    diagonale(2:n-1) = (2:n-1) * cos(x(2:n-1)) - (3:n) * sin(x(2:n-1)) + (1:n-2) * sin(x(2:n-1)); 
    Hess_x = diag(diagonale);
end

f = @(x) BTP_func(x);
gradf = @(x) BTP_grad(x);
Hessf = @(x) BTP_Hess(x);

% setting the values for the dimension
dimension = [1e1 1e2 1e3];
iter_max = 150;
tol = 1e-6;
avg_execution_time_NM = zeros(3,1);
failure_struct_NM = zeros(3,1); %for each dimension we count the number of failure

for dim = 1:3
    n = dimension(dim);
    x0 = ones(n,1);
    
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
    [xbest, ~,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],iter_max,tol);
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
        [xbest,~,iter,fbest, flag, failure] = nelderMead(f,x0_rndgenerated(:,i),[],[],[],[],iter_max,tol);
        time = time + toc(t1);
        xbest_struct(:,i+1) = xbest;
        iter_struct(1,i+1) = iter;
        fbest_struct(1,i+1) = fbest;

        % if failure = true (failure == 1), the run was unsuccessful; otherwise
        % failure = 0
        failure_struct_NM(dim) = failure_struct_NM(dim) + failure;
    end

    % I store the average execution time
    avg_execution_time_NM(dim,1) = time/11;

end

failure_struct_NM
avg_execution_time_NM

%% problem 75
clc
clear
close all

seed=min(339268,343310);
rng(seed);

% Definition of the problem
%n=1e4; %togli
F_75= @(x) 0.5*((x(1)-1)^2 + sum( (10*(1:length(x)-1)'.*(x(2:end)-x(1:end-1)).^2).^2 ) );
gradF_75= @(x) gradient_pb_75(x);
hessF_75= @(x) hessian_pb_75(x);
%h = 1e-2; %togli
approx_gradF_75= @(x) approxgradient_pb_75(x,h,'COST');
approx_hessF_75= @(x) approxhessian_pb_75(x,h,'COST');

%% RUNNING THE EXPERIMENTS ON NEALDER MEAD
format short e

iter_max = 200;
tol = 1e-12;
[rho, chi, gamma, sigma] = deal(1.1, 2.5, 0.6, 0.5);

% setting the dimensionality
dimension = [10 25 50];


% initializing the structures to store some stats for every dimension and
% starting point
execution_time_SX = zeros(length(dimension),11);
failures_SX = zeros(length(dimension),11); %we count the number of failures
iter_SX = zeros(length(dimension),11);
fbest_SX = zeros(length(dimension),11);
roc_SX = zeros(length(dimension),11);

for dim = 1:length(dimension)
    n = dimension(dim);
    x_esatto = ones(n,1);

    % defining the given initial point
    x0 = -1-2*ones(n,1);
    x0(end) = -1;

    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    % where [a, b] = [x0-1, x0+1]
    rng(seed);
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);

    % SOLVING SIMPLEX METHOD
    % first initial point
    t1 = tic;
    [~, xseq,iter,fbest, ~, failure] = nelderMead(F_75,x0,rho,chi,gamma,sigma,iter_max*size(x0,1),tol);
    execution_time_SX(dim,1) = toc(t1);
    fbest_SX(dim,1) = fbest;
    iter_SX(dim,1) = iter;
    roc_SX(dim,1) = compute_roc(xseq,x_esatto);
    disp(['**** SIMPLEX METHOD FOR THE PB 75 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

    disp(['Time: ', num2str(execution_time_SX(dim,1)), ' seconds']);

    disp('**** SIMPLES METHOD : RESULTS *****')
    disp('************************************')
    disp(['f(xk): ', num2str(fbest)])
    disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
    disp(['Rate of Convergence: ', num2str(roc_SX(dim,1))])
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
    failures_SX(dim,1) = failures_SX(dim,1) + failure;

    for i = 1:10
        t1 = tic;
        [~,~,iter,fbest, ~, failure] = nelderMead(F_75,x0_rndgenerated(:,i),rho,chi,gamma,sigma,iter_max*size(x0,1),tol);
        execution_time_SX(dim,i+1) = toc(t1);
        fbest_SX(dim,i+1) = fbest;
        iter_SX(dim,i+1) = iter;
        failures_SX(dim,i+1) = failures_SX(dim,i+1) + failure;
        roc_SX(dim,i+1) = compute_roc(xseq, x_esatto);

        disp(['**** SIMPLEX METHOD FOR THE PB 75 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

        disp(['Time: ', num2str(execution_time_SX(dim,i+1)), ' seconds']);
    
        disp('**** SIMPLES METHOD : RESULTS *****')
        disp('************************************')
        disp(['f(xk): ', num2str(fbest)])
        disp(['N. of Iterations: ', num2str(iter),'/',num2str(iter_max*size(x0,1))])
        disp(['Rate of Convergence: ', num2str(roc_SX(dim,i+1))])
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


varNames = ["average fbest", "average number of iterations", "average time of execution (sec)", "number of failures", "average rate of convergence"];
rowNames = string(dimension');
TSX = table(sum(fbest_SX,2)/11, sum(iter_SX,2)/11, sum(execution_time_SX,2)/11, sum(failures_SX,2),sum(roc_SX,2)/11 ,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TSX)

%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD 
% with exact gradient and hessian
format short e

% setting the values for the dimension and the parameters
dimension = [1e3 1e4 1e5];
max_iter_per_dimension = [1e3, 1e4, 1e5];
tol = 1e-6;
param = [0.4, 1e-4, 36; 0.3, 1e-4, 28; 0.4, 1e-3, 36];


% initializing structures to store some stats
execution_time_MN = zeros(length(dimension),11);
failure_struct_MN = zeros(length(dimension),11); % number of failures
iter_struct_MN = zeros(length(dimension),11);
fbest_struct_MN = zeros(length(dimension),11);
gradf_struct_MN = zeros(length(dimension),11);
roc_struct_MN = zeros(length(dimension),11);
ultima_direz_discesa = zeros(length(dimension), 11);

for dim = 1:length(dimension)
    n = dimension(dim);
    iter_max = max_iter_per_dimension(dim);
    [rho, c1, btmax] = deal(param(dim, 1), param(dim, 2), param(dim, 3));
    x_esatto = ones(n,1);

    %defining the given initial point
    x0 = -1.2*ones(n,1);
    x0(n) = -1;
    
    % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
    rng(seed);
    x0_rndgenerated = zeros(n,10);
    x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
    

    % SOLVING MODIFIED NEWTON METHOD METHOD
    % first initial point
    t1 = tic;
    [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, cos_pk_gradf] ...
        = modified_Newton(F_75,gradF_75, hessF_75, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', x_esatto);       
    execution_time_MN(dim,1) = toc(t1);
    fbest_struct_MN(dim,1) = fbest;
    iter_struct_MN(dim,1) = iter;
    gradf_struct_MN(dim,1) = gradfk_norm;
    roc_struct_MN(dim,1) = compute_roc(xseq, x_esatto);
    ultima_direz_discesa(dim,1) = cos_pk_gradf;
    disp(['**** MODIFIED NEWTON METHOD FOR THE PB 75 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);

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
            disp(['cosine of the angle between the last computed direction pk and the gradient: ', num2str(cos_pk_gradf)])
            disp(['last values of steplength alphak: ', mat2str((rho.^btseq(max(1:length(btseq)-10):end))')])
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
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, cos_pk_gradf] ...
            = modified_Newton(F_75,gradF_75, hessF_75, x0_rndgenerated(:,i), iter_max, rho, c1, btmax, tol, [], 'ALG', x_esatto);       
        execution_time_MN(dim,i+1) = toc(t1);
        fbest_struct_MN(dim,i+1) = fbest;
        iter_struct_MN(dim,i+1) = iter;
        failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
        gradf_struct_MN(dim,i+1) = gradfk_norm;
        roc_struct_MN(dim,i+1) = compute_roc(xseq, x_esatto);
        ultima_direz_discesa(dim,i+1) = cos_pk_gradf;

        disp(['**** MODIFIED NEWTON METHOD FOR THE PB 75 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);

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
                disp(['cosine of the angle between the last computed direction pk and the gradient: ', num2str(cos_pk_gradf)])
                disp(['last values of steplength alphak: ', mat2str((rho.^btseq(max(1:length(btseq)-10):end))')])


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

% % plotto cos_pk_gradf
% bar(ultima_direz_discesa')
% ylabel('cos(angolo)')
% title('Ultimo valore assunto da t(pk)*gradfk/(norm_pk * norm_gradfk)')
% legend({'dim = 1e3', 'dim = 1e4', 'dim = 1e5'}, "Box", 'on', 'Location', 'best')
% 

varNames = ["avg fbest", "avg gradf_norm","avg num of iters", "avg time of exec (sec)", "n failure", "avg roc"];
rowNames = string(dimension');
TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
format bank
display(TMN)


%% RUNNING THE EXPERIMENTS ON MODIFIED NEWTON METHOD 
% with approximated gradient and hessian
format short e

max_iter_per_dimension=[2*1e3, 2*1e4, 8*1e4];
tol = 1e-4;

% setting the values for the dimension
h_values = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]; 
dimension = [1e3, 1e4, 1e5];

param = [0.8, 1e-5, 90; 0.8, 1e-5, 90; 0.8, 1e-5, 90];
type_h = 'REL';

% initializing structures to store some stats
execution_time_MN_h = zeros(length(dimension),6);
failure_struct_MN_h = zeros(length(dimension),6); 
iter_struct_MN_h = zeros(length(dimension),6);
fbest_struct_MN_h = zeros(length(dimension),6);
gradf_struct_MN_h = zeros(length(dimension),6);
roc_struct_MN_h = zeros(length(dimension),6);


for id_h = 1:length(h_values)
    h = h_values(id_h);

    approx_gradF_75 = @(x) approxgradient_pb_75 (x,h,type_h);
    approx_hessF_75 = @(x) approxhessian_pb_75 (x,h,type_h);

    % initializing structures to store some stats
    execution_time_MN = zeros(length(dimension),11);
    failure_struct_MN = zeros(length(dimension),11); % number of failures
    iter_struct_MN = zeros(length(dimension),11);
    fbest_struct_MN = zeros(length(dimension),11);
    gradf_struct_MN = zeros(length(dimension),11);
    roc_struct_MN = zeros(length(dimension),11);
    ultima_direz_discesa = zeros(length(dimension), 11);
    
    
    for dim = 1:length(dimension)
        n = dimension(dim);
        x_esatto = ones(n,1);
        iter_max = max_iter_per_dimension(dim);
        [rho, c1, btmax] = deal(param(dim, 1), param(dim, 2), param(dim, 3));
    
        %defining the given initial point
        x0 = -1.2*ones(n,1);
        x0(n) = -1;
        
        % in order to generate random number in [a,b] I apply the formula r = a + (b-a).*rand(n,1)
        rng(seed);
        x0_rndgenerated = zeros(n,10);
        x0_rndgenerated(1:n, :) = x0(1:n) - 1 + 2.*rand(n,10);
        
    
        % SOLVING MODIFIED NEWTON METHOD METHOD
        % first initial point
        t1 = tic;
        [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, cos_pk_gradf] ...
            = modified_Newton(F_75,approx_gradF_75, approx_hessF_75, x0, iter_max, rho, c1, btmax, tol, [], 'ALG', x_esatto);       
        execution_time_MN(dim,1) = toc(t1);
        fbest_struct_MN(dim,1) = fbest;
        iter_struct_MN(dim,1) = iter;
        gradf_struct_MN(dim,1) = gradfk_norm;
        roc_struct_MN(dim,1) = compute_roc(xseq,x_esatto);
        ultima_direz_discesa(dim,1) = cos_pk_gradf;
        disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 75 (point ', num2str(1), ', dimension ', num2str(n), '):  *****']);
    
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
                disp(['cosine of the angle between the last computed direction pk and the gradient: ', num2str(cos_pk_gradf)])
                disp(['last values of steplength alphak: ', mat2str((rho.^btseq(max(1:length(btseq)-10):end))')])

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
            [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure, cos_pk_gradf] ...
                = modified_Newton(F_75,approx_gradF_75, approx_hessF_75, x0_rndgenerated(:,i), iter_max, rho, c1, btmax, tol, [], 'ALG', x_esatto);       
            execution_time_MN(dim,i+1) = toc(t1);
            fbest_struct_MN(dim,i+1) = fbest;
            iter_struct_MN(dim,i+1) = iter;
            failure_struct_MN(dim,i+1) = failure_struct_MN(dim,i+1) + failure;
            gradf_struct_MN(dim,i+1) = gradfk_norm;
            roc_struct_MN(dim,i+1) = compute_roc(xseq,x_esatto);
            ultima_direz_discesa(dim,i+1) = cos_pk_gradf;
    
            disp(['**** MODIFIED NEWTON METHOD WITH FIN DIFF ( ', type_h, ' with h = ', num2str(h), ') FOR THE PB 75 (point ', num2str(i+1), ', dimension ', num2str(n), '):  *****']);
    
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
                    disp(['cosine of the angle between the last computed direction pk and the gradient: ', num2str(cos_pk_gradf)])
                    disp(['last values of steplength alphak: ', mat2str((rho.^btseq(max(1:length(btseq)-10):end))')])

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
    
% plotto cos_pk_gradf
bar(ultima_direz_discesa')
ylabel('cos(angolo)')
title('Ultimo valore assunto da t(pk)*gradfk/(norm_pk * norm_gradfk)')
legend({'dim = 1e3', 'dim = 1e4', 'dim = 1e5'}, "Box", 'on', 'Location', 'best')
    
    
    varNames = ["avg fbest", "avg gradf_norm","avg num of iters", "avg time of exec (sec)", "n failure", "avg roc"];
    rowNames = string(dimension');
    TMN = table(sum(fbest_struct_MN,2)/11, sum(gradf_struct_MN,2)/11 ,sum(iter_struct_MN,2)/11, sum(execution_time_MN,2)/11, sum(failure_struct_MN,2), sum(roc_struct_MN,2)/11,'VariableNames', varNames, 'RowNames', rowNames);
    format bank
    display(TMN)

    execution_time_MN_h(:,id_h) = sum(execution_time_MN,2)/11;
    failure_struct_MN_h(:,id_h) = sum(failure_struct_MN,2); 
    iter_struct_MN_h(:,id_h) = sum(iter_struct_MN,2)/11;
    fbest_struct_MN_h(:,id_h) = sum(fbest_struct_MN,2)/11;
    gradf_struct_MN_h(:,id_h) = sum(gradf_struct_MN,2)/11;
    roc_struct_MN_h(:,id_h) = sum(roc_struct_MN,2)/11;
end


%% 
% Running the problem on Nelder_Mead method 
tol=1e-6;
% n=50;
x0= -1.2*ones(n,1);
x0(n)= -1;
x_esatto = ones(n,1);
% F_75= @(x) 0.5*((x(1)-1)^2 + sum( (10*(1:length(x)-1)'.*(x(2:end)-x(1:end-1)).^2).^2 ) );

% rho=1.5; %tune
% chi=2.5; %tune
% gamma=0.6; %tune
% sigma=0.5; %tune
% 
% [~, xseq,iter,fbest, ~, failure] = nelderMead(F_75,x0,rho,chi,gamma,sigma,200*n,tol);
% roc = compute_roc(xseq,x_esatto)

% I draw the graph for n=2
% Function definition for n=2
F = @(x1, x2) 0.5 * ((x1 - 1).^2 + (10 * (2 - 1) * (x2 - x1).^2).^2);

x1_range = linspace(-2, 2, 100); 
x2_range = linspace(-2, 2, 100); 
[X1, X2] = meshgrid(x1_range, x2_range);
Z = F(X1, X2);

figure;
surf(X1, X2, Z, 'EdgeColor', 'none'); 
colormap jet; 
colorbar;
xlabel('x_1');
ylabel('x_2');
zlabel('F(x)');

% Running the problem on modified nuewton method with exact gradient and
% hessian
%n=1e4;
rho=0.6; %tune
c1=1e-3; %tune
% tic
% [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] ...
%     = modified_Newton(F_75,gradF_75, hessF_75, x0, 50*n, rho, c1, 150, [], [], [], ones(n,1));
% time=toc
% rate_of_convergence = compute_roc(xseq,ones(n,1))

% Running the problem on modified newton method with approximated gradient
% and hessian
% tic
% [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] ...
%     = modified_Newton(F_75,approx_gradF_75, approx_hessF_75, x0, 10*n, rho, c1, 65, [], [], [], ones(n,1));
% time=toc
% rate_of_convergence = compute_roc(xseq,ones(n,1))

%prove, poi togli
% h=1e-12;
% x=[-1;1;-1;1];
% x=2*ones(4,1);
% veraH = hessF_75(x)
% approssimataH = approxhessian_pb_75(x,h,'REL')
% he_k=zeros(4,1);
% he_k(1)=h;
% (F_75(x+2*he_k)-2*F_75(x+he_k)+F_75(x))/(h^2)
% ((-2+2*h)^2 + (10*((1+1-2*h)^2))^2 - 2*((-2+h)^2) -2*((10*(1+1-h)^2)^2) + (-2)^2 + (10*((2)^2))^2 )/(2*(h^2))


%functions definition of exact gradient
function grad = gradient_pb_75 (x)
    n=length(x);
    grad=zeros(n,1);
    grad(1)= x(1)-1-200*(x(2)-x(1))^3;
    grad(2:n-1)= 200*((1:n-2)'.^2 .* (x(2:n-1)-x(1:n-2)).^3 - (2:n-1)'.^2 .* (x(3:n)-x(2:n-1)).^3 );
    grad(n)= 200*(n-1)^2*(x(n)-x(n-1))^3;
end

%function definition of exact hessian
function hess = hessian_pb_75(x)
    n=length(x);
    diag_princ=zeros(n,1); %d^2F/dx_k^2
    diag_princ(1)=1+600*(x(2)-x(1))^2;
    diag_princ(2:n-1)= 600* ((1:n-2)'.^2 .* (x(2:n-1) - x(1:n-2)).^2 + (2:n-1)'.^2 .* (x(3:n) - x(2:n-1)).^2 );
    diag_princ(n)= 600 * (n-1)^2 * (x(n)-x(n-1))^2;
    diag_upper=zeros(n,1);
    diag_upper(2:n)= -600* ( (1:n-1)'.^2 .* (x(2:n)-x(1:n-1)).^2 );
    diag_upper=diag_upper(2:n); %long n-1
    %matrix whose columns are the diags of the sparse hessian
    D=[[diag_upper;0], diag_princ, [0;diag_upper]];
    hess=spdiags(D,-1:1,n,n);
end

%function definition of approximated gradient (using forward differences)
function grad = approxgradient_pb_75 (x,h,type_h)
    n= length(x);
    grad= zeros(n,1);
    for i=1:n
        switch type_h
            case 'COST'
                hi = h;
            case 'REL'
                hi = h*abs(x(i));
        end
        if i==1
            grad(i) = ((x(1)+hi-1)^2 ...
                + (10*(x(2)-x(1)-hi)^2)^2 ...
                - (x(1)-1)^2 - (10*(x(2)-x(1))^2)^2)/(2*hi) ;
        elseif i<n
            %(F(x+he_i) - F(x))/hk =
            %f_i^2(x+he_i)+f_{i+1}^2(x+he_i)-f_i^2(x)-f_{i+1}^2(x)
            grad(i) = ((10*(i-1)*(x(i)+hi-x(i-1))^2)^2 ...
                + (10*(i)*(x(i+1)-x(i)-hi)^2)^2 ...
                - (10*(i-1)*(x(i)-x(i-1))^2)^2 - (10*i*(x(i+1)-x(i))^2)^2)/(2*hi) ;
        else %i==n
            grad(i) = ((10*(n-1)*(x(n)+hi-x(n-1))^2)^2 - (10*(n-1)*(x(n)-x(n-1))^2)^2)/(2*hi);
        end
        
    end
end

%function definition of approximated hessian (using forward differences)
function hess = approxhessian_pb_75 (x,h,type_h)
    n = length(x);
    % In this case I know that the hessian is sparse (tridiagonal). The
    % total number of non-zero elements in principle is n + 2*(n-1) = 3*n-2
    indices_i = zeros(3*n-2, 1);
    indices_j = zeros(3*n-2, 1);
    values = zeros(3*n-2, 1);
    iter = 1; 

    for k=1:n
        switch type_h
            case 'COST'
                hk = h;
                hkm1 = h;
            case 'REL'
                hk = h*abs(x(k));
                if k>1
                    hkm1 = h*abs(x(k-1));
                end

        end

        %diagonal element
        indices_i(iter) = k;
        indices_j(iter) = k;
        %(f(x+2*he_k)-2*f(x+he_k)+fx)/(hk^2)=(f_k^2(x+2he_k)+f_{k+1}^2(x+2he_k)
        %-2f_k^2(x+he_k)-2f_{k+1}^2(x+he_k)+f_k^2(x)+f_{k+1}^2(x))/(2hk^2)
        if k==1
            values(iter) = ((x(1)+2*hk-1)^2 + (10*(x(2)-x(1)-2*hk)^2)^2 ...
                -2*(x(1)+hk-1)^2 -2*(10*(x(2)-x(1)-hk)^2)^2 ...
                + (x(1)-1)^2 + (10*(x(2)-x(1))^2)^2 )/(2*hk^2);
            iter = iter+1;
        elseif k<n
            values(iter) = ((10*(k-1)*(x(k)+2*hk-x(k-1))^2)^2 + (10*(k)*(x(k+1)-x(k)-2*hk)^2)^2 ...
                -2*(10*(k-1)*(x(k)+hk-x(k-1))^2)^2 -2*(10*(k)*(x(k+1)-x(k)-hk)^2)^2 ...
                + (10*(k-1)*(x(k)-x(k-1))^2)^2 + (10*(k)*(x(k+1)-x(k))^2)^2 )/(2*hk^2);
            iter = iter+1;
        else %k==n
            values(iter) = ((10*(n-1)*(x(n)+2*hk-x(n-1))^2)^2 ...
                -2*(10*(n-1)*(x(n)+hk-x(n-1))^2)^2 ...
                +(10*(n-1)*(x(n)-x(n-1))^2)^2 )/(2*hk^2);
            iter = iter+1;
        end

        %lower diagonal element (only exists if k>1)
        if k>1 && k<n
            indices_i(iter) = k; 
            indices_j(iter) = k-1;
            values(iter) = ((10*(k-1)*(x(k)+hk - x(k-1)-hkm1)^2)^2 ...
                    + (10*k*(x(k+1)-x(k)-hk)^2)^2 ...
                    - (10*(k-1)*(x(k)+hk-x(k-1))^2)^2 ...
                    - (10*k*(x(k+1)-x(k)-hk)^2)^2 ...
                    - (10*(k-1)*(x(k)-x(k-1)-hkm1)^2)^2 ...
                    - (10*k*(x(k+1)-x(k))^2)^2 ...
                    + (10*(k-1)*(x(k)-x(k-1))^2)^2 ...
                    + (10*k*(x(k+1)-x(k))^2)^2)/(2*hk*hkm1);
            iter = iter+1; 

            %for simmetry, upper diagonal element
            indices_i(iter) = k-1; 
            indices_j(iter) = k;
            values(iter) = values(iter-1);
            iter = iter+1;

        elseif k==n
            indices_i(iter) = k; 
            indices_j(iter) = k-1;
            values(iter) = ((10*(k-1)*(x(k)+hk - x(k-1)-hkm1)^2)^2 ...
                    - (10*(k-1)*(x(k)+hk-x(k-1))^2)^2 ...
                    - (10*(k-1)*(x(k)-x(k-1)-hkm1)^2)^2 ...
                    + (10*(k-1)*(x(k)-x(k-1))^2)^2)/(2*hk*hkm1);
            iter = iter+1; 

            %for simmetry, upper diagonal element
            indices_i(iter) = k-1; 
            indices_j(iter) = k;
            values(iter) = values(iter-1);
            iter = iter+1;
        end
    end
    hess = sparse(indices_i, indices_j, values, n, n);
end


%to compute the rate of convergence
function rate_of_convergence = compute_roc(xseq, x_esatto)
if size(xseq,2) >=4 && isempty(x_esatto)
    k = size(xseq,2) -1;
    norm_ekplus1 = norm(xseq(:, k+1) - xseq(:,k));
    norm_ek = norm(xseq(:, k) - xseq(:,k-1));
    norm_ekminus1 = norm(xseq(:, k-1) - xseq(:,k-2));
    rate_of_convergence = log(norm_ekplus1/norm_ek) / log(norm_ek/norm_ekminus1);
elseif size(xseq,2)>=3 && ~isempty(x_esatto)
    norm_ekplus1 = norm(x_esatto - xseq(:,end));
    norm_ek = norm(x_esatto - xseq(:,end-1));
    norm_ekminus1 = norm(x_esatto - xseq(:,end-2));
    rate_of_convergence = log(norm_ekplus1/norm_ek) / log(norm_ek/norm_ekminus1);
else 
    rate_of_convergence = nan;
end
end

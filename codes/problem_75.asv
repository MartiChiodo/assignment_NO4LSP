%% problem 75
clc
clear
close all

seed=min(339268,343310);
rng(seed);

%definition of the problem
F_75= @(x) 0.5*((x(1)-1)^2 + sum( (10*(1:length(x)-1)'.*(x(2:end)-x(1:end-1)).^2).^2 ) );
gradF_75= @(x) gradient_pb_75(x);
hessF_75= @(x) hessian_pb_75(x);
h = 1e-8;
approx_gradF_75= @(x) approxgradient_pb_75(x,h,'COST');
approx_hessF_75= @(x) approxhessian_pb_75(x,h,'COST');

%running the problem on modified nuewton method with exact gradient and
%hessian
n=1e4;
x0= -1.2*ones(n,1);
x(n)= -1;
rho=0.8; %tune
c1=1e-2; %tune
% tic
% [xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] ...
%     = modified_Newton(F_75,gradF_75, hessF_75, x0, 50*n, rho, c1, [], [], [], [], ones(n,1));
% time=toc
% rate_of_convergence = compute_roc(xseq)

%running the problem on modified newton method with approximated gradient
%and hessian
tic
[xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] ...
    = modified_Newton(F_75,approx_gradF_75, approx_hessF_75, x0, 50*n, rho, [], 150, [], [], [], ones(n,1));
time=toc
rate_of_convergence = compute_roc(xseq)

%functions definition of exact gradient
function grad = gradient_pb_75 (x)
    n=length(x);
    grad=zeros(n,1);
    grad(1)= x(1)-1-200*(x(2)-x(1))^3;
    %for k=2:n-1
    %    grad(k)=200*((k-1)^2 * (x(k)-x(k-1))^3 - k^2 * (x(k+1)-x(k))^3 );
    %end
    grad(2:n-1)= 200*((1:n-2)'.^2 .* (x(2:n-1)-x(1:n-2)).^3 - (2:n-1)'.^2 .* (x(3:n)-x(2:n-1)).^3 );
    grad(n)= 200*(n-1)^2*(x(n)-x(n-1))^3;
end

%function definition of exact hessian
function hess = hessian_pb_75(x)
    n=length(x);
    diag_princ=zeros(n,1); %d^2F/dx_k^2
    diag_princ(1)=1;
    diag_princ(2:n-1)= 600* ((1:n-2)'.^2 .* (x(2:n-1) - x(1:n-2)).^2 + (2:n-1)'.^2 .* (x(3:n) - x(2:n-1)).^2 );
    diag_princ(n)= 600 * (n-1)^2 * (x(n)-x(n-1))^2;
    diag_upper=zeros(n,1);
    diag_upper(2:n)= -600* ( (1:n-1)'.^2 .* (x(2:n)-x(1:n-1)).^2 );
    diag_upper=diag_upper(2:n); %long n-1
    %matrix whose columns are the diags of the sparse hessian
    D=[[diag_upper;0], diag_princ, [0;diag_upper]];
    hess=spdiags(D,-1:1,n,n);
    hess = (hess + hess')/2; %to guarantee simmetry 
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
            case 'REL'
                hk = h*abs(x(k));
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
        if k>1
            indices_i(iter) = k; 
            indices_j(iter) = k-1;
            %(f(x+he_k+he_km1) - f(x+he_k) - f(x+he_km1) + fx)/(hk^2) =
            %(2f_k^2(x)-f_k^2(x+h_ek)-f_k^2(x+he_k-1))/(2hk^2)
            values(iter) = (2*(10*(k-1)*(x(k)-x(k-1))^2)^2 ...
                -(10*(k-1)*(x(k)+hk-x(k-1))^2)^2 ...
                -(10*(k-1)*(x(k)-x(k-1)-hk)^2)^2 )/(2*hk^2);
            iter = iter+1; 

            %for simmetry, upper diagonal element
            indices_i(iter) = k-1; 
            indices_j(iter) = k;
            values(iter) = values(iter-1);
            iter = iter+1;
        end

%         %lower diagonal element (only exists if k<n)
%         if k<n
%             indices_i(iter) = k; 
%             indices_j(iter) = k+1;
%             he_kp1 = zeros(n,1);
%             he_kp1(k+1) = hk;
%             values(iter) = (f(x+he_k+he_kp1) - f(x+he_k) - f(x+he_kp1) + fx)/(hk^2);
%             iter = iter+1; 
%         end
    end
    hess = sparse(indices_i, indices_j, values, n, n);
    

%     diag_princ = zeros(n,1);
%     diag_upper = zeros(n-1,1);
%     diag_lower = zeros(n-1,1);
%     % In this case I know that the hessian is sparse (tridiagonal).
%     % We use the approximation of the hessian as the jacobian of the gradient
%     % using the approximated gradient (VA BENE FARLO COSì O SI PUò FARE SOLO
%     % COL GRADIENTE ESATTO??)
%     gradfx = gradf(x);
%     
%     %gradf(x+he_1+he_4+he_7+...)
%     hei_mod1 = zeros(n,1);
%     hei_mod1(1:3:n) = h;
%     gradfx_hei_mod1 = gradf(x + hei_mod1);
%     
%     %approximation of columns 1,4,7,... of the hessian
%     cols_mod1 = (gradfx_hei_mod1 - gradfx)./h;
%     diag_princ(1:3:end) = cols_mod1(1:3:end);
%     diag_lower(1:3:end) = cols_mod1(2:3:end);
%     diag_upper(3:3:end) = cols_mod1(3:3:end);
%     
%     %gradf(x+he_2+he_5+he_8+...)
%     hei_mod2 = zeros(n,1);
%     hei_mod2(2:3:n) = h;
%     gradfx_hei_mod2 = gradf(x + hei_mod2);
%     
%     %approximation of columns 2,5,8,... of the hessian
%     cols_mod2 = (gradfx_hei_mod2 - gradfx)./h;
%     diag_princ(2:3:end) = cols_mod2(2:3:end);
%     diag_lower(2:3:end) = cols_mod2(3:3:end);
%     diag_upper(1:3:end) = cols_mod2(1:3:end-1);
%     
%     %gradf(x+he_3+he_6+he_9+...)
%     hei_mod0 = zeros(n,1);
%     hei_mod0(3:3:n) = h;
%     gradfx_hei_mod0 = gradf(x + hei_mod0);
%     
%     %approximation of columns 3,6,9,... of the hessian
%     cols_mod0 = (gradfx_hei_mod0 - gradfx)./h;
%     diag_princ(3:3:end) = cols_mod0(2:3:end);
%     diag_lower(3:3:end) = cols_mod0(3:3:end);
%     diag_upper(2:3:end) = cols_mod0(1:3:end-1);
%     
%     D=[[diag_lower;0], diag_princ, [0;diag_upper]];
%     hess=spdiags(D,-1:1,n,n);
end


%to compute the rate of convergence
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

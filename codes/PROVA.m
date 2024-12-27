%% PROVEEEE
clear all
close all
clc

% Parametric Rosenbrock function in dimension n 
function f = parametric_rosenbrock(x, alpha)
    f = 0;
    n = length(x);
    for i = 2:n
        f = f + alpha * (x(i) - x(i-1)^2)^2 + (x(i-1)-1)^2;
    end
end

function gradf = grad_parametric_rosenbrock(x,alpha)
    n = length(x);
    gradf = zeros(n,1);
    
    for k = 2:n-1
        gradf(k,1) = -2*alpha*x(k-1)^2 + x(k)*(2*alpha +2) -2  +4*alpha*x(k)^3- 4*alpha*x(k)*x(k+1);
    end

    gradf(1,1) = 2*(x(1) -1) + 4*alpha*x(1)^3 - 4*alpha*x(1)*x(2);
    gradf(n,1) = -2*alpha*x(n-1)^2 + 2*alpha*x(n) ;

end

function Hessf = hess_parametric_rosenbrock(x,alpha)
    n = length(x);
    diags = zeros(n,3);
    % diags(:,1) is the principal one, diags(:,2) is the superior one and
    % diags(:,3) is the inferior one

    diags(1,1) = 2 + 12*alpha*x(1)^2 - 4*alpha*x(2);
    diags(n,1) = 2*alpha;
    diags(n-1,3) = -4*alpha*x(n-1);
    diags(n,2) = -4*alpha*x(n-1);

    for k = 2:n-1
       diags(k,1) = 2*alpha + 12*alpha*x(k)^2 - 4*alpha*x(k+1) +2;
       diags(k-1,3) = -4*alpha*x(k-1); %diag inferior: k is the first derivative
       diags(k,2)= -4*alpha*x(k-1); %diag superior: k id the first derivative
    end
    

    Hessf = spdiags(diags, [0, +1, -1], n, n);

end


% the excercice asks to fix alpha = 100
f = @(x) parametric_rosenbrock(x, 100);
gradf = @(x) grad_parametric_rosenbrock(x,100);
Hessf = @(x) hess_parametric_rosenbrock(x,100);

n = 1e4;
% minimum point is such that gradf(x) = 0
x_esatto = fsolve(gradf, zeros(n,1), '');
f(x_esatto)

%% vediamo se il minimo varia all'aumentare della dimensione o se è costante
x = -15:0.1:15;
y = -15:0.1:15;
[X, Y] = meshgrid(x, y);
Z = arrayfun(@(x, y) f([x; y]), X, Y);

% Grafico 3D
f1 = figure;
surf(X, Y, Z);
title('Grafico 3D di f(x, y)');
xlabel('x');
ylabel('y');
zlabel('f(x, y)');
shading interp; 

% % curve di livello
% contour(X, Y, Z, 20); % '20' indica il numero di livelli
% colorbar; % Aggiunge una barra colori per riferimento
% xlabel('x');
% ylabel('y');
% title('Curve di livello della funzione f');


%% PROVA MODIFIED NEWTON METHOD
tol = 1e-4;
% x0 = (1:1:n)'; % pto iniziale es3_marti
% x0 = ones(n,1); % pto iniziale Rosenbrock
% x0(1:2:n) = -1.2;
n = 1e3;
x0 = -ones(n,1); %pto iniziale pb 79

rho = 0.9;  c1 = 1e-3; btmax = 150; tau_kmax = 1e4; 
[xbest_MN, xseq_MN, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] ...
    = modified_Newton(f,gradf, Hessf, x0, 5000, rho, c1, btmax, tol, tau_kmax)


%% PROVANEALDER MEAD
n = 10;
% x0 = (1:1:n)'; % pto iniziale es3_marti
% x0 = ones(n,1); % pto iniziale Rosenbrock
% x0(1:2:n) = -1.2;
x0 = -ones(n,1); %pto iniziale pb 79

[xbest, xseq,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],n*500,[])



%%

clear all
clc
close all

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

n = 1e3;

% minimum point is such that gradf(x) = 0
x_esatto = fsolve(gradf, zeros(n,1), '');
f(x_esatto)



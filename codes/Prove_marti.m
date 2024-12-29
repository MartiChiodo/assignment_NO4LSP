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

%% forma della funzione
x = -0.5:0.001:0.5;
y = -0.5:0.001:0.5; 
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
% contour(X, Y, Z, 60); % '20' indica il numero di livelli
% colorbar; % Aggiunge una barra colori per riferimento
% xlabel('x');
% ylabel('y');
% title('Curve di livello della funzione f');


%% PROVA MODIFIED NEWTON METHOD
tol = 1e-4;


n = 1e5;
% x0 = 2*ones(n,1); %pto iniziale pb 76
% x0 = 0.5*ones(n,1); %pto iniziale pb 82
x0 = ones(n,1); % pto iniziale Rosenbrock
x0(1:2:n) = -1.2;
% x0 = ones(n,1); % pto iniziale pb 60
% x0(1:2:n) = 0;
% x0 = ones(n,1); %pro iniziale pb 64

rho = 0.3;  c1 = 1e-4; btmax = 38;  % per 1e3
% rho = 0.5;  c1 = 1e-3; btmax = 48;  % per 1e4
% rho = 0.4;  c1 = 1e-3; btmax = 36; % per 1e5
[~, ~, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] ...
    = modified_Newton(f,gradf, Hessf, x0, 5000, rho, c1, btmax, tol, [], 'ALG', ones(n,1))


%% PROVANEALDER MEAD
n = 50;
% x0 = (1:1:n)'; % pto iniziale es3_marti
% x0 = ones(n,1); % pto iniziale Rosenbrock
% x0(1:2:n) = -1.2;
% x0 = 2 *ones(n,1); %pto iniziale pb 76
x0 = ones(n,1); %pro iniziale pb 64

[xbest, xseq,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],n*500,[])



%%

n = 1e3;

% minimum point is such that gradf(x) = 0
options = optimoptions('fsolve', 'FunctionTolerance', 1e-15, 'OptimalityTolerance', 1e-10);
[x,fval,exitflag,output]  = fsolve(gradf, 0.5*ones(n,1),  options)
f(x)


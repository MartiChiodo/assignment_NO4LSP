%% PROVEEEE
clear all
close all
clc

function Fx = PF1_funct(x)
    % x is a matrix, each col contains a vector of dimension n
    % Fx is a vector, the i-th element is F(x(:,i))

    Fx = zeros(1,size(x,2));
    size(x,1)
    for col = 1:size(x,2)
        Fx(1,col) = 0.5* 1e-5 * sum((x(:,col) - ones(size(x,1),1)).^2) + 0.5*(sum(x(:,col).^2) - 0.25)^2;
    end
end

f = @(x) PF1_funct(x);
gradf = @(x) x .* (2+1e-5) -1e-5 + (sum(x.^2) -0.25).*x;

function hessf = hessian(x)
    n = length(x);
    diags = zeros(n,3); %1st column is the principal diag, 2nd column is the superior diag and 3rd column is the inferior
    diags(1:n,1) = 1e-5 + 4*x(1:n).^2 + 2*(sum(x(:,1).^2) -0.25);
    diags(2:n,3) = 4.*x(1:n-1).*x(2:n);
    diags(1:n-1,2) = 4.*x(2:n).*x(1:n-1);
    hessf = spdiags(diags, [0,1,-1], n,n);
end

Hessf = @(x) hessian(x);

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


%% prova NEALDER MEAD MATLAB
n = 1e3;
x0 = (1:1:n)';
options = optimset('Display','iter','PlotFcns',@optimplotfval);
[x,fval,exitflag,output] = fminsearch(f,x0,options)
x0

%% PROVA MODIFIED NEWTON METHOD
tol = 1e-7;
n = 1e2;
% x0 = (1:1:n)'; % pto iniziale es3_marti
x0 = ones(n,1); % pto iniziale Rosenbrock
x0(1:2:n) = -1.2;

rho = 0.5; c1 = 1e-4; btmax = 45; tau_kmax = 1e4; 
[xbest_MN, xseq_MN, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] = modified_Newton(f,gradf, Hessf, x0, 1000, rho, c1, btmax, tol, tau_kmax);


%% PROVANEALDER MEAD
n = 50;
% x0 = (1:1:n)'; % pto iniziale es3_marti
x0 = ones(n,1); % pto iniziale Rosenbrock
x0(1:2:n) = -1.2;

[xbest, xseq,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],n*500,[])



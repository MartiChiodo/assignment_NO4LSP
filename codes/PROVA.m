%% PROVEEEE
clear all
close all
clc

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



%% vediamo se il minimo varia all'aumentare della dimensione o se è costante
x = -5:0.1:5;
y = -5:0.1:5;
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




%% prova NEALDER MEAD MATLAB
n = 1e3;
x0 = (1:1:n)';
options = optimset('Display','iter','PlotFcns',@optimplotfval);
[x,fval,exitflag,output] = fminsearch(f,x0,options)
x0

%% PROVA MODIFIED NEWTON METHOD
tol = 1e-7;
n = 1e5;
x0 = (1:1:n)';
rho = 0.5; c1 = 1e-4; btmax = 50; tau_kmax = 1e4; 
[xbest_MN, xseq_MN, iter_MN, fbest_MN, gradfk_norm_MN, btseq_MN, flag_bcktrck_MN, failure_MN] = modified_Newton(f,gradf, Hessf, x0, 1000, rho, c1, btmax, tol, tau_kmax);


%% NEALDER MEAD
n = 1e2;
x0 = (1:1:n)';
[xbest, xseq,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],n*500,[]);
iter
failure
xbest

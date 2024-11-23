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
Hessf = @(x) (2+1e-5)*eye(length(x));


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


% punti stazionari --> dim n
dimensioni = (2:1:100);
minimi = zeros(length(dimensioni),1);
for n = dimensioni
    k_vector = -5:1:5;
    x0 = zeros(n,length(k_vector));
    for i = 1:length(k_vector)
        k = k_vector(i);
        x0(1,i) = pi*k - atan(2);
        for row = 2:n-1
            x0(row,i) = pi*k -atan(2/row);
        end
        x0(n,i) = pi*k + atan((n-1)/n);
    end
    minimi(n-1,1) = min(f(x0));
end

dimensioni = transpose(dimensioni);
f2 = figure;
plot(dimensioni,minimi, '.');
title('Valore del minimo al variare della dimensione');
xlabel('dimensione');
ylabel('min(f)');


%% prova NEALDER MEAD MATLAB
n = 100;
x0 = (1:1:n)';
options = optimset('Display','iter','PlotFcns',@optimplotfval);
%[x,fval,exitflag,output] = fminsearch(f,x0,options)
x0
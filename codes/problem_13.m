%% generalization of the Brown function 2
clc
clear
close all

F= @(x) sum(x(1:2:(length(x)-1)).^(2*x(2:2:length(x)).^2+2) ...
    +x(2:2:length(x)).^(2*x(1:2:(length(x)-1)).^2+2) );

%F= @(x) sum(x(1:2:(length(x)-1)).^2.^(x(2:2:length(x)).^2+1) ...
    %+x(2:2:length(x)).^2.^(x(1:2:(length(x)-1)).^2+1) );

%prova n=2
x0=[-2;1];
%F(x0)

seed=339268;
rng(seed);
[xbest,iter,fbest, flag]= nelderMead(F,x0,[],[],[],[],10,[])
%con 10 iter restituisce un fbest complesso ??? 0.0714 + 0.0016i

%% n=100
n=100;
x0=-1*ones(n,1);
x0(2:2:n)=1;
%simplex0=zeros(n,n+1);
%    simplex0(:,1)=x0;
%    for i=1:n
%        ei=zeros(n,1);
%        ei(i)=5; %provo a partire con un simplesso più grande
%        simplex0(:,i+1)=x0+ei;
%    end
%    x0=simplex0;
[xbest,iter,fbest, flag]= nelderMead(F,x0,[],[],[],[],100,[])
%anche cambiando la condizione di terminazione, fa quasi solo reflection

%con quello di matlab
options = optimset('MaxIter',100,'TolX',1e-6);
[X,FVAL,EXITFLAG] = fminsearch(F,x0,options)
%sembra sia ancora peggio

%% togli, solo per capire com'è fatta
F = @(x1, x2) x1.^2 .* (x2.^2 + 1) + x2.^2 .* (x1.^2 + 1);

% Crea una griglia di punti tra -5 e 5
x1 = linspace(-5, 5, 100); % 100 punti per x1
x2 = linspace(-5, 5, 100); % 100 punti per x2
[X1, X2] = meshgrid(x1, x2); % Genera la griglia 2D

% Calcola i valori della funzione sulla griglia
Z = F(X1, X2);

% Crea un grafico 3D della funzione
figure;
surf(X1, X2, Z); % Superficie 3D
xlabel('x_1');
ylabel('x_2');
zlabel('F(x_1, x_2)');



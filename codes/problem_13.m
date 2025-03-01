
%% generalization of the Brown function 2
clc
clear
close all

%F= @(x) sum(abs(x(1:2:end-1)).^(2*x(2:2:end).^2+2) ...
    %+abs(x(2:2:end)).^(2*x(1:2:end-1).^2+2) );
%F= @(x) sum(x(1:2:(length(x)-1)).^2.^(x(2:2:length(x)).^2+1) ...
    %+x(2:2:length(x)).^2.^(x(1:2:(length(x)-1)).^2+1) );

F = @(x) sum((x(1:2:end-1).^2).^(2 * (x(2:2:end).^2 + 1)) + ...
             (x(2:2:end).^2).^(2 * (x(1:2:end-1).^2 + 1)));

%prova n=2
x0=[-2;1];
%F(x0)

seed=339268;
rng(seed);
[xbest,xseq,iter,fbest, flag, failure]= nelderMead(F,x0,[],[],[],[],[],[])

%% nelder mead
n=10;
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
tic
[xbest,xseq,iter,fbest, flag, failure]= nelderMead(F,x0,[],[],[],[],[],[]);
toc
fbest
iter

%n=1e3 toc=761.918316 seconds.
%n=1e4 toc=156950.744228 seconds
%      iter= 65003
%      fbest=5.937149363053800e-04
%      xbest=-1.552657772845642e-02
%             1.572342098828969e-02 (alternati)
ooc=log(norm(xseq(:,end))/norm(xseq(:,end-1)))/log(norm(xseq(:,end-1))/norm(xseq(:,1)));
%      ooc=8.510933382627016e-01

%% con quello di matlab
%options = optimset('MaxIter',100,'TolX',1e-6);
tic
[X,FVAL,EXITFLAG] = fminsearch(F,x0);
toc
FVAL
%sembra sia ancora peggio

%n=1e3 toc=548.955172 seconds. MA NON TROVA LA SOLUZIONE


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

%% newton modificato
n=2;
itermax=100;
x0=-1*ones(n,1);
x0(2:2:n)=1;

F = @(x) sum((x(1:2:end-1).^2).^(2 * (x(2:2:end).^2 + 1)) + ...
             (x(2:2:end).^2).^(2 * (x(1:2:end-1).^2 + 1)));

gradF_pb_13 = @(x) gradient_pb_13(x);
HessF_pb_13 = @(x) Hessian_pb_13 (x);

[xbest, xseq, iter, fbest, gradfk_norm, btseq, flag_bcktrck, failure] =...
    modified_Newton(F,gradF_pb_13, HessF_pb_13, x0,itermax , [], [], [], [], 1e5)


function gradF = gradient_pb_13 (x)
n=length(x);
gradF=zeros(n,1);
gradF(2:2:end)=(x(1:2:end).^2).^(x(2:2:end).^2+1).*log(x(1:2:end).^2)*2.*x(2:2:end) ...
    +2*(x(1:2:end).^2+1).*x(2:2:end).^(2*x(1:2:end).^2+1);
gradF(1:2:end)=(x(2:2:end).^2).^(x(1:2:end).^2+1).*log(x(2:2:end).^2)*2.*x(1:2:end) ...
    +2*(x(2:2:end).^2+1).*x(1:2:end).^(2*x(2:2:end).^2+1);
end

function HessF = Hessian_pb_13 (x)
n=length(x); 
diag_princ=zeros(n,1);
diag_princ(2:2:end)= (x(1:2:end).^2).^(x(2:2:end).^2+1).*log(x(1:2:end).^2).^2.*4.*(x(2:2:end).^2) ...
    +2.*(x(1:2:end).^2).^(x(2:2:end).^2+1).*log(x(1:2:end).^2)+ ...
    2.*(x(1:2:end).^2+1).*(2*x(1:2:end).^2+1).*x(2:2:end).^(2.*x(1:2:end).^2); %k even
diag_princ(1:2:end)= (x(2:2:end).^2).^(x(1:2:end).^2+1).*log(x(2:2:end).^2).^2.*4.*x(1:2:end).^2 ...
    +2.*(x(2:2:end).^2).^(x(1:2:end).^2+1).*log(x(2:2:end).^2)+ ...
    2.*(x(2:2:end).^2+1).*(2.*x(2:2:end).^2+1).*x(1:2:end).^(2.*x(2:2:end).^2); %k odd
D(:,2)= diag_princ; %d^2F/dx_k^2
diag_upper=zeros(n,1);
diag_upper(1:2:end)= 4*x(2:2:end).*(x(1:2:end).^(2*x(2:2:end).^2+1)) ...
    +2*(x(2:2:end).^2+1).*x(1:2:end).^(2*x(2:2:end).^2+1).*log(x(1:2:end))*4.*x(2:2:end) ...
    +2*(x(1:2:end).^2+1).*x(2:2:end).^(2*x(1:2:end).^2+1).*log(x(2:2:end).^2)*2.*x(1:2:end) ...
    +4*x(2:2:end).^(2*x(1:2:end).^2+1).*x(1:2:end);
%VA TOLTO L'ULTIMO PER AVERLO LUNGO n-1?
diag_upper=diag_upper(1:end-1,:);
%matrix whose columns are the diags of the sparse hessian
D=[[0; diag_upper], diag_princ, [diag_upper; 0]];

HessF=spdiags(D,-1:1,n,n);
end




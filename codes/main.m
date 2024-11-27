%% ESERCIZIO 2

clear all
clc

format long e

% Rosenbrock function in dimension n=2
f= @(x) 100*(x(2)-x(1)^2)^2+(1-x(1))^2; 
x0=[1.2;1.2];

[xbest,xseq,iter,fbest, flag, failure] = nelderMead(f,x0,[],[],[],[],50,1e-9)

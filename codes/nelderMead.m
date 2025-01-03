function [xbest,xseq,iter,fbest, flag, failure]= nelderMead(f,x0,rho,chi,gamma,sigma,kmax,tol)

close all

% [xbest,iter,fbest]= nelderMead(f,x0,rho,chi,gamma,sigma,kmax,tol)
% 
% Functiopn that finds the minimizer of the function f using the Nealder
% Method.
%
% INPUTS:
% f = function handle that return the value of the function we want to minimize f : R^n --> R
% x0 = either the initial point (x in R^n) of the method or the initial
% symplex (x in R^(n,n+1), the columns of x are the vertices of the symplex)
% rho = reflection factor
% chi = expansion factor
% gamma = contraction factor
% sigma = shrinking factor
% kmax = maximum number of iterations
% tol = tollerance on the absolute value of f(xN) - f(x1)
%
% OUTPUTS:
% xbest = the last xk computed by the function
% xseq = matrix n x 3, the k-th col contains the best point of the symplex
% of the last 3 iterates
% which minimize the function f
% iter = number of iterations
% fbest = approximation of the minimum
% flag = if true, the given symplex was degenere
% failure = if true, we are declaring a failure
%

% we are verifying that all the parameters are passed as inputs, eventually
% we set rho, chi, gamma and sigma with default valuess
if isempty(rho)
    rho=1;
end
if isempty(chi)
    chi=2;
end
if isempty(gamma)
    gamma=0.5;
end
if isempty(sigma)
    sigma=0.5;
end
if isempty(kmax)
    kmax=200*size(x0,1);
end
if isempty(tol)
    tol=1e-6;
end


n=size(x0,1); %dimension of the space we are working
flag = false;
failure = false;

if rank(x0) < n && size(x0,2) > 1
    % se il simplesso è degenere ritorniamo flag = true
    flag = true;
    xbest = nan; iter = 0; fbest = nan;
    return
end


% preliminary analysis of the function in order to compute a smart complex
eval_pt = [1, 50, -50, 100, -100, 300, -300, 600, -600, 1000, -1000, 3000, -3000, 6000, -6000 10000, -10000];
best_direction = zeros(n,1);
for comp = 1:n
    eval = zeros(length(eval_pt),1);
    id = 1;
    for pt = eval_pt
        x = x0;
        x(comp) = x(comp) + pt;
        eval(id) = f(x);
        id = id + 1;
    end

    [~, id_pt] = min(eval);
    best_direction(comp) = eval_pt(id_pt);
end

disp("ho finito la valutazione di funzione")


if size(x0,2)==1 
    %se in input c'è un solo punto costruiamo il simplesso di partenza 
    simplex0=zeros(n,n+1);
    simplex0(:,1)=x0;
    for i=1:n
        ei=zeros(n,1);
        ei(i) = best_direction(i);
        % ei(i) = 0.05*x(i); % according to Matlab implementation
        simplex0(:,i+1)= x0 + ei;
    end 
    x0=simplex0;
end


fk=zeros(n+1,1);
comp=0;

% sorting the point based on the evaluation of the function in the point
for i=1:n+1
    fk(i)=f(x0(:,i));
end
[fk_sorted,indices]=sort(fk);


xseq = zeros(n,4);
cont = 1;
xseq(:,cont) = x0(:,indices(1));
best_values = []; % list I will use to plot the convergence of the method


while comp<kmax && (fk_sorted(n) - fk_sorted(1)) > tol 
    shrinking=false; %false se devo aggiornare solo un punto, true se ho fatto shrink

    % indices last element
    np1 = indices(end);

    % we are keeping the n best vertices to compute the centroid
    x0_best_n=x0;
    x0_best_n(:,indices(end))=[]; 
    centroid=sum(x0_best_n(:, 1:n))/n; 
    centroid = centroid';
    
    % REFLECTION PHASE
    xR= (1+rho) * centroid - rho * x0(:,np1);
    fxR=f(xR);
    if fxR>fk_sorted(1) && fxR<fk_sorted(n)
        xnew=xR;
        %x0(:,indices(end))=xnew; %se non lo metto non lo aggiorna (con il continue passa subito all'iterazione successiva?)
        %continue
    elseif fxR<=fk_sorted(1)
        % EXPANSION PHASE
        xE= (1+rho*chi) * centroid - rho*chi*x0(:, np1);
        if f(xE)<fxR
            xnew=xE;
            %x0(:,indices(end))=xnew; %se non lo metto non lo aggiorna (con il continue passa subito all'iterazione successiva?)
            %continue
        else
            xnew=xR;
            %x0(:,indices(end))=xnew; %?
            %continue
        end
    else
        % CONTRACTION PHASE
        if fxR > fk_sorted(n+1)
            % inside contraction
            xC=(1-gamma) * centroid + gamma*x0(:,np1);
        else
            % outside contraction
            xC= (1 + gamma*rho)*centroid - gamma * rho*x0(:, np1);
        end
        if f(xC)<fk_sorted(end)
            xnew=xC;
        else
            % SHRINKING PHASE
            shrinking=true;
            x=zeros(n,n+1);
            x(:,1:n+1)=x0(:,indices(1))+sigma.*(x0(:,1:n+1)-x0(:,indices(1)));
            x(:,indices(1))=x0(:,indices(1));
            x0=x;
        end
    end

    % if we have not shrunk the symplex, we have to update the symplex by
    % replacing the worst vertice with the new one
    if ~shrinking
        x0(:,np1)=xnew;
    end

    % PREPARATION for next iterations
    comp=comp+1;

    % sorting the point based on the evaluation of the function in the point
    for i=1:n+1
        fk(i)=f(x0(:,i));
    end
    [fk_sorted,indices]=sort(fk);

    % updating xseq
    if cont == 4
        cont = 1;
    else
        cont = cont + 1;
    end
    xseq(:,cont) = x0(:,indices(1));

    best_values(end+1) = fk_sorted(1);
    % plot
    if mod(comp, 10) == 0
        figure(1);
        plot(best_values, '-o', 'MarkerSize', 4);
        xlabel('Iterations');
        ylabel('Best Evaluation');
        title('Progress minimum value Nealder Mead');
        drawnow;
    end

end

% computing the minimizer and the minimum found
xbest = x0(:,indices(1));
iter = comp;
fbest = fk_sorted(1);

% cutting xseq and ordering in in such a way that the last column is the
% most recent solution
m = min(iter,4); %number of iterations available in xseq
xseq = xseq(:,1:m);
shift = mod(cont,m);
xseq = circshift(xseq,-shift,2);


if iter == kmax && (fk_sorted(n) - fk_sorted(1)) > tol 
    failure = true;
end





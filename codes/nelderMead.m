function [xbest,xseq,iter,fbest, flag, failure]= nelderMead(f,x0,rho,chi,gamma,sigma,kmax,tol)

% [xbest,iter,fbest]= nelderMead(f,x0,rho,chi,gamma,sigma,kmax,tol)
% 
% Functiopn that finds the minimizer of the function f using the Nealder
% Method.
%
% INPUTS:
% f = function we want to minimize f : R^n --> R
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
% xbest = the approximation of the minimizer
% xseq = matrix n x iter, the k-th col contains the point of the symplex
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
    kmax=200*size(x,1);
end
if isempty(tol)
    tol=1e-6;
end


%arguments
%    f
%    x0 %matrice nxn+1 con n+1 colonne ognuna delle quali è un vertice del simplesso di partenza
%    rho double =1
%    chi double =2
%    gamma double =0.5
%    sigma double =0.5
%    kmax integer =50
%    tol double =1e-6
%end

%aggiungere controllo su simplesso degenere (nel caso in cui x0 sia un
%simplesso)


n=size(x0,1); %dimension of the space we are working
flag = false;
failure = false;

if rank(x0) < n && size(x0,2) > 1
    % se il simplesso è degenere ritorniamo flag = true
    flag = true;
    xbest = nan; iter = 0; fbest = nan;
    return
end

if size(x0,2)==1 
    %se in input c'è un solo punto costruiamo il simplesso di partenza 
    simplex0=zeros(n,n+1);
    simplex0(:,1)=x0;
    for i=1:n
        ei=zeros(n,1);
        ei(i)=1;
        simplex0(:,i+1)=x0+ei;
    end
    x0=simplex0;
end


fk=zeros(n+1,1);
k=0;

% sorting the point based on the evaluation of the function in the point
for i=1:n+1
    fk(i)=f(x0(:,i));
end
[fk_sorted,indices]=sort(fk);

xseq = zeros(n,kmax+1);
xseq(:,1) = x0(:,indices(1));

while k<kmax && (fk_sorted(n) - fk_sorted(1)) > tol 
    shrinking=false; %false se devo aggiornare solo un punto, true se ho fatto shrink

    % we are keeping the n best vertices to compute the centroid
    x0_best_n=x0;
    x0_best_n(:,indices(end))=[]; 
    centroid=mean(x0_best_n,2); 
    
    % REFLECTION PHASE
    xR=centroid + rho*(centroid-x0(:,indices(end)));
    fxR=f(xR);
    if fxR>=fk_sorted(1) && fxR<fk_sorted(n)
        xnew=xR;
        %x0(:,indices(end))=xnew; %se non lo metto non lo aggiorna (con il continue passa subito all'iterazione successiva?)
        %continue
    elseif fxR<fk_sorted(1)
        % EXPANSION PHASE
        xE=centroid+chi.*(xR-centroid);
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
            xC=centroid-gamma.*(centroid-x0(:,indices(end)));
        else
            xC=centroid-gamma.*(centroid-xR);
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
        x0(:,indices(end))=xnew;
    end

    % PREPARATION for next iterations
    k=k+1;

    % sorting the point based on the evaluation of the function in the point
    for i=1:n+1
        fk(i)=f(x0(:,i));
    end
    [fk_sorted,indices]=sort(fk);

    xseq(:,k+1) = x0(:,indices(1));

end

% computing the minimizer and the minimum found
xbest = x0(:,indices(1));
iter = k;
fbest = fk_sorted(1);

% cutting xseq
xseq = xseq(:,1:iter+1);

if iter == kmax && (fk_sorted(n) - fk_sorted(1)) > tol 
    failure = true;
end





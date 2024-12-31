function [JFx] = findiff_J(F, x, h, type)
%
% function [JFx] = findiff_J(F, x, h, type)
%
% Function that approximate the Jacobian of F in x (column vector) with the
% finite difference (forward/centered) method.
%
% INPUTS:
% F = function handle that describes a function R^n->R^m;
% x = n-dimensional column vector;
% h = the h used for the finite difference computation of gradf
% type = 'fw' or 'c' for choosing the forward/centered finite difference
% computation of the gradient. If the thype is 'fw_sym' or 'c_sym', the
% output matrix is symmetric.
%
% OUTPUTS:
% JFx = matrix m-by-n corresponding to the approximation
% of the Jacobian of F in x.

m = length(F(x));
n = length(x);
JFx = zeros(m, n);


switch type
    case {'fw','fw_sym'}
        for i=1:n
            xh = x;
            xh(i) = xh(i) + h;
            JFx(:, i) = (F(xh) - F(x)) / h;
        end
    case {'c', 'c_sym'}
        for i=1:n
            xh_plus = x;
            xh_minus = x;
            xh_plus(i) = xh_plus(i) + h;
            xh_minus(i) = xh_minus(i) - h;
            JFx(:, i) = (F(xh_plus) - F(xh_minus)) / (2 * h);
        end
    otherwise % same 'fw' case
        for i=1:n
            xh = x;
            xh(i) = xh(i) + h;
            JFx(:, i) = (F(xh) - F(x)) / h;
        end
end

if isequal(type, 'fw_sym') || isequal(type, 'c_sym')
        JFx = 0.5 * (JFx + JFx');
end

end


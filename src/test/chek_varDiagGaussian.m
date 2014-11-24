function chek_varDiagGaussian()
%CHEK_VARDIAGGAUSSIAN chek_varDiagGaussian()
%   Gradient check for varDiagGaussian().
clear functions
dim = 5;
mu = rand(dim,1);
x = mu + 0.01*rand(dim,1);
s = rand(dim,1);
theta = [mu; s];
[mygrad, delta] = gradchek(theta', @f, @grad, x);
disp('maxdiff = ')
disp(max(abs(delta)))
end

function val = f(theta, x)
dim = numel(x);
mu = theta(1:dim)';
s = theta(dim+1:end)';
val = varDiagGaussian(x,mu,s);
end

function d = grad(theta, x)
dim = numel(x);
mu = theta(1:dim)';
s = theta(dim+1:end)';
[~,dm,ds] = varDiagGaussian(x,mu,s);
d = [dm;ds]';
end
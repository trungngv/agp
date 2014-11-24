function chek_varGaussian()
%CHEK_VARGAUSSIAN chek_varGaussian()
%   Gradient check for varGaussian().
clear functions
dim = 5;
mu = rand(dim,1);
x = mu + 0.01*rand(dim,1);
A = rand(dim,dim);
Sigma = A*A';
Lambda = jit_chol(Sigma)';
theta = [mu(:); Lambda(:)];
[mygrad, delta] = gradchek(theta', @f, @grad, x);
disp('maxdiff = ')
disp(max(abs(delta)))
end

function val = f(theta, x)
dim = numel(x);
mu = theta(1:dim)';
Lambda = reshape(theta(numel(x)+1:end),dim,dim);
Lambda = tril(Lambda);
[~,~,val] = varGaussian(x,mu,Lambda);
end

function d = grad(theta, x)
dim = numel(x);
mu = theta(1:dim)';
Lambda = reshape(theta(numel(x)+1:end),dim,dim);
Lambda = tril(Lambda);
[dmu,dLambda] = varGaussian(x,mu,Lambda);
d = [dmu(:);dLambda(:)]';
end
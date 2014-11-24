function testCholeskyParametrization()
dim = 5;
mu = rand(dim,1);
x = mu + 0.1*rand(dim,1);
A = rand(dim,dim);
Sigma = A*A';
Lambda = jit_chol(Sigma);
Lambda = setDiag(Lambda, log(diag(Lambda)));
% Note: if we use cholesky parametrization then the lower triangular of
% Lambda must not be treated as parameters. Don't know an efficient way of
% doing so in Matlab though..
theta = [mu(:); Lambda(:)];
[theta, fX] = minimize(theta, @f, 100, x);
plot(fX)
end

function [val, grad] = f(theta,x)
dim = numel(x);
mu = theta(1:dim);
Lambda = reshape(theta(numel(x)+1:end),dim,dim);
[dmu,dLambda,val] = varGaussianLog(x,mu,Lambda);
grad = -[dmu(:); dLambda(:)];
val = -val;
end

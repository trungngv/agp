function [dmu,dLambda,f] = varGaussian(x,mu,Lambda)
%VARGAUSSIAN [dmu,dLambda,f] = varGaussian(x,mu,Lambda)
%
% Gradients of the log density of a multivariate Gaussian distribution with
% Sigma = Lambda Lambda^T, i.e. Lambda = chol(Sigma)'.
%   dmu = d log \Normal(x; mu, Sigma) / dmu 
%   dLambda = d log \Normal(x; mu, Sigma) / dLambda
% Note that Lambda must be a lower triangular matrix.
%
% INPUT
%   - x : 
% OUTPUT
invLambda = inv(Lambda);
y = invLambda*(x-mu);
% log N(x; mu, Sigma) = -log|Lambda| -0.5 (x-mu)' (Lamda Lambda')^(-1) (x-mu)
% f = -sum(log(diag(Lambda))) - 0.5*(y')*y;

% Lambda should always be lower triangular (for e.g. after stochastic
% update), in which cse use sum(log(diag(Lambda)) for log | Lambda|
% here we use log(det()) for checking gradient!
f = -log(det(Lambda)) - 0.5*(y'*y);
dmu = invLambda'*y;
dLambda = tril(-invLambda' + dmu*y');
end

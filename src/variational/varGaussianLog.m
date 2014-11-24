function [dmu,dLambda,f] = varGaussianLog(x,mu,Lambda)
%VARGAUSSIAN [dmu,dLambda,f] = varGaussianLog(x,mu,Lambda)
%
% Gradients of the log density of a multivariate Gaussian distribution with
% Sigma = Lambda Lambda^T, i.e. Lambda = chol(Sigma).
% dmu = d log \Normal(x; mu, Sigma) / dmu 
% dLambda = d log \Normal(x; mu, Sigma) / dLambda
%   
% INPUT
%   - x : 
% OUTPUT
Lambda = setDiag(Lambda, exp(diag(Lambda)));
y = Lambda\(x-mu);
% log N(x; mu, Sigma) = -log|Lambda| -0.5 (x-mu)' (Lamda Lambda')^(-1) (x-mu)
f = -sum(log(diag(Lambda))) - 0.5*(y')*y;
% -log(det(Lambda)) for gradient checks
% f = -log(det(Lambda)) - 0.5*(y'*y);
invLambda = inv(Lambda);
dmu = invLambda'*y;
dLambda = -invLambda' + dmu*y';
dLambda = setDiag(dLambda, diag(dLambda).*diag(Lambda));
end

function [mu,sigma2] = mixtureMeanAndVariance(muk,sigma2k)
%MIXTUREMEANANDVARIANCE [mu,sigma2] = mixtureMeanAndVariance(muk,sigma2k)
%   
% The mean and variance of a mixture distribution given the mean and
% variance of the mixture components. 
% If the mixture distribution is
%    p(y) = \sum_k p_k(y | ...)
% and the component distributions p_k(y | ...) have known mean mu_k
% and variance sigma_k then
%   E[y] = 1/K \sum mu_k
%   Var[y] = 1/K \sum sigma2_k + 1/K (mu_k)^2 - (1/K \sum mu_k)^2
mu = mean(muk,2);
sigma2 = mean(sigma2k,2) + mean(muk.^2,2) - mu.^2;



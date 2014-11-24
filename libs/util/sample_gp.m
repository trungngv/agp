function f = sample_gp(x, covfunc, hyp, N)
%SAMPLE_GP  f = sample_gp(x, covfunc, hyp, N)
%   
%  Generate a sample from a GP with given covariance function.
% 
% INPUT
%   - x : inputs
%   - covfunc : a covariance function in gpml package
%   - hyp : covaraince hyperparameters
%   - N : number of samples
%
% OUTPUT
%   - f : size(x,1) x N, each column is a sample
if iscell(covfunc)
  K = feval(covfunc{:}, hyp, x);
else
  K = feval(covfunc, hyp, x);
end  
L = jit_chol(K)';
U = randn(size(x,1),N);
f = L*U;
end


function [logllh,dhyp] = llhGaussian(y,f,hyp)
%LLHGAUSSIAN logllh = llhGaussian(y,f,hyp)
%   The standard Gaussian likelihood. We just use the GPML implementation
%   because both y and f are just vector for this model.
% 
% All of the likelihood models should implement with this signature.
%   y : N x P vector of outputs (P > 1 for multi-output)
%   f : N x Q matrix of latent function values
% The log likelihood log p(y_n | f_(n)) is custom implementation of each model.
logllh = likGauss(hyp.lik,y,f);
s2 = exp(2*hyp.lik);
if nargout == 2
  dhyp = sum((y-f).^2/s2 - 1);
end
end


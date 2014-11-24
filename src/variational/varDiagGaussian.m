function [fval,dm,ds] = varDiagGaussian(f,m,s)
%VARDIAGGAUSSIAN [fval,dm,ds] = varDiagGaussian(f,m,s)
%
% Gradients of the log density of a multivariate Gaussian distribution with
% diagonal covariance matrix. The implementation is efficient for this
% special case.
% 
% INPUT
%   - x
%   - m : the mean
%   - s : the log square root of the covariance diagonal log(sqrt(s^2))
% 
% OUTPUT
%   - fval
%   - dm, ds
%
N = size(f,1);
s2 = exp(2*s);

f0 = f - m;
fval = -0.5*(N*log(2*pi) + sum(log(s2)) + sum(f0.*f0./s2));
if nargout >= 2
  dm = f0./s2;
  ds = -1 + f0.*f0./s2;
end
end

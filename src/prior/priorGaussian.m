function [logprior,dloghyp] = priorGaussian(f,x,hyp)
%PRIORGAUSSIAN [logprior,dloghyp] = priorGaussian(f,x,hyp)
%   
% log \Normal(f; 0, K(x,x)) and its derivatives wrt covariance
% hyperparameters
persistent L
persistent Linv
persistent lasthyp
if isempty(L) || (any(lasthyp ~= hyp.cov))
  lasthyp = hyp.cov;
  K = feval(hyp.covfunc,lasthyp,x);
  L = jit_chol(K)'; % note the transpose
  Linv = inv(L);
end
invLf = Linv*f;
logprior = -sum(log(diag(L))) - 0.5*(invLf'*invLf);
if nargout == 2
  dloghyp = zeros(size(hyp.cov));
  % dlogN / dtheta = 0.5*trace((alpha alpha' - Kinv) dK/dtheta))
  % where alpha = Kinv*f
  alpha0 = Linv'*(invLf*invLf')*Linv - Linv'*Linv;
  for i=1:numel(dloghyp)
    dK = feval(hyp.covfunc,lasthyp,x,[],i);
    dloghyp(i) = 0.5*trAB(alpha0, dK);
  end
end
end


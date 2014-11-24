function [fval,dM,dL] = entropyFull(M,L,Q)
%ENTROPYFULL [fval,dM,dL] = entropyFull(M,L,Q)
%   The lower bound of entropy (Lent) for (factorized) full Gaussian posterior 
%   and its derivatives wrt variational parameters.
%
%   M : the means of the posterior
%   L : (QxN)*N, the j-th NxN matrix is the cholesky parameterization of
%   latent function j.
%
%   fval = 0.5 log |S| = \sum_j 0.5 log |Sj| = \sum_j log |Lj|

dM = zeros(size(M));
dL = zeros(size(L));
N = size(L,2);
% S = LL'
fval = 0;
for j=1:Q
  start_row = (j-1)*N+1;
  end_row = j*N;
  Lj = L(start_row:end_row,:);
  fval = fval + sum(log(diag(Lj)));
  if nargout > 1
    dLj = zeros(size(Lj));
    dL(start_row:end_row,:) = setDiag(dLj,1./diag(Lj));
  end
end
end


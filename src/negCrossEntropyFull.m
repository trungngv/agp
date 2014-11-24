function [fval,dM,dL,dhyp] = negCrossEntropyFull(M,L,hyp,x)
%NEGCROSSENTROPYFULL [fval,dM,dL,dhyp] = negCrossEntropyFull(M,L,hyp,x)
%   The negative cross entropy (Lcross) for (factorized) full Gaussian posterior 
%   and its derivatives wrt variational parameters.
%
%   Usage:
%     fval = negCrossEntropyFull(M,L,hyp,x)
%     [fval, dM, dL] = negCrossEntropyFull(M,L,hyp,x)
%     [fval, ~, ~, dhyp] = negCrossEntropyFull(M,L,hyp,x)
%
%   M : the means 
%   L : the Cholesky factors 
%   hyp.covfunc : covariance function
%   hyp.cov{j} : covariance hyperparameters of the j-th latent function
%   x : observations
N = size(x,1);
Q = size(M,1)/N;
% S = LL'
Lchol = cell(Q,1);
sumlogdet = 0;
for j=1:Q
  K = feval(hyp.covfunc, hyp.cov{j}, x);
  Lchol{j} = jit_chol(K);
  %Kinv{j} = invChol(Lchol{j});
  sumlogdet = sumlogdet + 2*sum(log(diag(Lchol{j})));
end
fval = Q*N*log(2*pi) + sumlogdet;
if nargout == 3
  dM = zeros(size(M));
  dL = zeros(size(L));
elseif nargout == 4
  dM = []; dL = []; dhyp = cell(Q,1);
end
for j=1:Q
  s_row = (j-1)*N+1;
  e_row = j*N;
  Lj = L(s_row:e_row,:);
  %fval = fval + M(s_row:e_row)'*Kinv{j}*M(s_row:e_row) + trAB(Kinv{j}, Lj*Lj');
  Kinvm = solve_chol(Lchol{j},M(s_row:e_row)); % K^{-1}m
  KinvLj = solve_chol(Lchol{j},Lj); % K^{-1} Lj
  fval = fval + M(s_row:e_row)'*Kinvm + trAB(KinvLj,Lj');
  if nargout == 3   % derivatives of variational parameters
    dM(s_row:e_row) = -Kinvm;
    dL(s_row:e_row,:) = -tril(KinvLj);
  elseif nargout == 4   % hyperparameters derivatives
    dhyp{j} = zeros(size(hyp.cov{j}));
    Kjinv = invChol(Lchol{j});
    for i=1:numel(hyp.cov{j})
      dK = feval(hyp.covfunc, hyp.cov{j}, x, [], i);
      dhyp{j}(i) = 0.5*trAB(Kinvm*Kinvm' - Kjinv + KinvLj*KinvLj', dK);
    end
    % use numerically more stable gradient for covSEard
    if strcmp(func2str(hyp.covfunc),'covSEard')
      dhyp{j}(i) = M(s_row:e_row)'*Kinvm - N + trAB(KinvLj,Lj');
    end
  end
end
fval = -0.5*fval;
end


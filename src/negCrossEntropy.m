function [fval,dM,dL,dw,dhyp] = negCrossEntropy(M,L,w,hyp,x)
%NEGCROSSENTROPY [fval,dM,dL,dw,dhyp] = negCrossEntropy(M,L,w,hyp,x)
%   The negative cross entropy (Lcross) and its derivatives wrt variational
%   parameters.
%
%   Usage:
%      fval = negCrossEntropy(M,L,w,hyp,x)
%      [fval, dM, dL, dw] = negCrossEntropy(M,L,w,hyp,x)
%      [fval, ~, ~, ~, dhyp] = negCrossEntropy(M,L,w,hyp,x)
%
%   M(:,k) : the means of the k-th component
%   L(:,k) : the log square root of the covaraince diagonal of the k-th component
%   w(k) : the log weight of the k-th component
%   hyp.covfunc : covariance function
%   hyp.cov{j} : covariance hyperparameters of the j-th latent function
%   x : observations
Km = size(M,2);
N = size(x,1);
Q = size(M,1)/N;
w = exp(w);
S = exp(2*L);
Lchol = cell(Q,1);
Kinv = cell(Q,1);
sumlogdet = 0;
for j=1:Q
  K = feval(hyp.covfunc, hyp.cov{j}, x);
  Lchol{j} = jit_chol(K);
  Kinv{j} = invChol(Lchol{j});
  sumlogdet = sumlogdet + 2*sum(log(diag(Lchol{j})));
end
fval = repmat(Q*N*log(2*pi) + sumlogdet,Km,1);
if nargout == 4
  dM = zeros(size(M));
  dL = zeros(size(S));
  dw = zeros(size(w));
elseif nargout == 5
  dM = []; dL = []; dw = []; dhyp = cell(Q,1);
  for j=1:Q
    dhyp{j} = zeros(numel(hyp.cov{j}),1);
  end
end
% assuming all covfuncs take the same number of hyperparameters
dK = cell(Q,numel(hyp.cov{1}));
for k=1:Km
  for j=1:Q
    s_row = (j-1)*N+1;
    e_row = j*N;
    Kinvm = solve_chol(Lchol{j},M(s_row:e_row,k));
    Skj = S(s_row:e_row,k);
    fval(k) = fval(k) + M(s_row:e_row,k)'*Kinvm + sum(diag(Kinv{j}).*Skj);
    if nargout == 4
      dM(s_row:e_row,k) = 2*Kinvm;
      dL(s_row:e_row,k) = 2*diag(Kinv{j}).*Skj;
    elseif nargout == 5
      for i=1:numel(hyp.cov{j})
        if isempty(dK{j,i})
          dK{j,i} = feval(hyp.covfunc, hyp.cov{j}, x, [], i);
        end
        dthetai = 0.5*w(k)*trAB(Kinvm*Kinvm' - Kinv{j} + Kinv{j}*diagAB(Skj,Kinv{j}), dK{j,i});
        dhyp{j}(i) = dhyp{j}(i) + dthetai;
      end
      if strcmp(func2str(hyp.covfunc),'covSEard')
        dhyp{j}(i) = dhyp{j}(i) - dthetai + w(k)*(-N + M(s_row:e_row,k)'*Kinvm...
          + sum(diag(Kinv{j}).*Skj));
      end
    end
  end
  if nargout == 4
    dM(:,k) = -0.5*w(k)*dM(:,k);
    dL(:,k) = -0.5*w(k)*dL(:,k);
    dw(k) = -0.5*w(k)*fval(k);
  end
end
fval = -0.5*sum(w.*fval);
end


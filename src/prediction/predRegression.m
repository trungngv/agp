function [fmu,fvar,yvar] = predRegression(m,conf,xstar)
%PREDREGRESSION  [fmu,fvar,yvar] = predRegression(m,conf,xstar)
%   Prediction by a regression model with single latent function.
Q = m.Q; N = m.N;
fmu = []; fvar = []; yvar = [];
s_rows = (0:(Q-1))'*N + 1;
e_rows = (1:Q)'*N;
if ~isfield(conf,'latentnoise')
  sn2 = 0;
else
  sn2 = conf.latentnoise;
end
for j=1:Q
  M = m.pars.M(s_rows(j):e_rows(j),:);
  if isfield(m.pars,'S')
    % the full gaussian with efficient parametrization case, S is given
    S = m.pars.S{j};
  else
    % the mixture case (L is diagonal) and blackbox case (L is block)
    S = m.pars.L(s_rows(j):e_rows(j),:)*m.pars.L(s_rows(j):e_rows(j),:)';
  end
  covhyp = m.pars.hyp.cov{j};
  likhyp = m.pars.hyp.lik;
  if nargout == 3
    [fmuj,fvarj,yvarj] = predOne(M,S,m.x,xstar,conf.covfunc,covhyp,likhyp,sn2);
    yvar = [yvar; yvarj];
  else
    [fmuj,fvarj] = predOne(M,S,m.x,xstar,conf.covfunc,covhyp,likhyp,sn2);
  end
  fmu = [fmu; fmuj];
  fvar = [fvar; fvarj];
end
end

% prediction for one latent function
function [fmu,fvar,yvar] = predOne(M,S,x,xstar,covfunc,covhyp,likhyp,sn2)
Kff = feval(covfunc, covhyp, x) + sn2*eye(size(x,1));
Lff = jit_chol(Kff,3);
invKff = invChol(Lff);

Kss = feval(covfunc, covhyp, xstar, 'diag') + sn2;
Kfs = feval(covfunc, covhyp, x, xstar);
fmu =  Kfs'*(invKff*M);

% we can also compute full covariance at a higher cost
% diag(Ksm * kmminv * S * Kmmonv *Kms) 
var_1 =  sum(Kfs.*(invKff*S*invKff*Kfs),1)';
var_2 =  sum(Kfs.*(invKff*Kfs),1)';
fvar = var_1 + Kss - var_2;
fvar = max(fvar,1e-10); % remove numerical noise i.e. negative variance
if nargout == 3
  yvar = fvar + exp(2*likhyp(end));
end
end


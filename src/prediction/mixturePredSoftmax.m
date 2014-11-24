function [pstar,pall] = mixturePredSoftmax(m,conf,xstar,ystar)
%MIXTUREPREDSOFTMAX [p,pall] = mixturePredSoftmax(m,conf,xstar,ystar)
%   Prediction by a multi-class classification model with mixture of Gaussian posteriors.
%
ntest = size(xstar,1);
pstar = zeros(ntest,1);
pall = zeros(size(ystar));
for k=1:m.K
  comp = m;
  comp.pars.M = m.pars.M(:,k);
  comp.pars.L = diag(exp(m.pars.L(:,k)));
  [pk,pallk] = predSoftmax(comp,conf,xstar,ystar);
  pstar = pstar + pk;
  pall = pall + pallk;
end
pstar = pstar/m.K;
pall = pall/m.K;


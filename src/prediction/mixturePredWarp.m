function [lp, ymean, yvar] = mixturePredWarp(m,conf,xstar,ystar)
%MIXTUREPREDWARP [lp,ymean,yvar] = mixturePredWarp(m,conf,xstar,ystar)
%   Prediction by a warped model with mixture of Gaussian posteriors.
%
lp = zeros(size(xstar,1),m.K);
ymean = lp;
yvar = lp;
for k=1:m.K
  comp = m;
  comp.pars.M = m.pars.M(:,k);
  comp.pars.L = diag(exp(m.pars.L(:,k)));
  [lp(:,k),ymean(:,k),yvar(:,k)] = predWarp(comp,conf,xstar,ystar);
end
lp = mean(lp,2);
[ymean,yvar] = mixtureMeanAndVariance(ymean,yvar);



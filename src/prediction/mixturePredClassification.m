function [lp, ymean, yvar] = mixturePredClassification(m,conf,xstar,ystar)
%MIXTUREPREDCLASSIFICATION  [lp,fmu,fvar] = mixturePredClassification(m,conf,xstar,ystar)
%   Prediction by a classification model with mixture of Gaussian posteriors.
%
if nargin == 3
  ystar = [];
end
lp = zeros(size(xstar,1),m.K);
ymean = lp;
yvar = lp;
for k=1:m.K
  comp = m;
  comp.pars.M = m.pars.M(:,k);
  comp.pars.L = diag(exp(m.pars.L(:,k)));
  [lp(:,k), ymean(:,k), yvar(:,k)] = predClassification(comp,conf,xstar,ystar);
end
lp = mean(lp,2);
[ymean,yvar] = mixtureMeanAndVariance(ymean,yvar);



function [fmuout,fvar,yvar] = mixturePredRegression(m,conf,xstar)
%MIXTUREPREDREGRESSION  [fmu,fvar,yvar] = mixturePredRegression(m,conf,xstar)
%   Prediction by a regression model with mixture of Gaussian posteriors.
fmu = zeros(size(xstar,1),m.K);
fvar = fmu;
yvar = fmu;
for k=1:m.K
  comp = m;
  comp.pars.M = m.pars.M(:,k);
  comp.pars.L = diag(exp(m.pars.L(:,k)));
  [fmu(:,k),fvar(:,k),yvar(:,k)] = predRegression(comp,conf,xstar);
end
[fmuout,fvar] = mixtureMeanAndVariance(fmu,fvar);
[~,yvar] = mixtureMeanAndVariance(fmu,yvar);


function [lp, ymu, ys2] = predClassification(m,conf,xstar,ystar)
%PREDCLASSIFICATION [lp, ymu, ys2] = predClassification(m,conf,xstar,ystar)
%   Prediction by the binary classification model.
%   
% INPUT
%   m : learned model
%   conf: model configurations
%   xstar : test inputs
%
% OUTPUT
%   p(y*=1|xstar,y)
[fmu, fvar] = predRegression(m,conf,xstar);
if nargin == 3 || isempty(ystar)
  ystar = ones(size(xstar,1),1);
end
%[lp, ymu, ys2] = likErf([], yones, fmu, fvar);
[lp, ymu, ys2] = likLogistic([], ystar, fmu, fvar);
end


function [nlpd,nlpds] = mynlpd(ytrue, ymu, yvar)
%MYNLPD [nlpd,nlpds] = mynlpd(ytrue, ymu, yvar)
%   Computes the negative log predictive (for a Gaussian predictive
%   distribution) of the test points.
% 
% INPUT
%   - ytrue : true output values
%   - ymu : predictive means
%   - yvar : predictive variances
%
% OUTPUT
%   - nlpds = -log p(ytrue_t; ymu_t, yvar_t) for all t in test set
%   - nlpd = mean(nlpds) the average negative log predictive density 
%
% Trung V. Nguyen
% 12/05/14
nlpds = 0.5*(ytrue-ymu).^2./yvar+log(2*pi*yvar);
nlpd = mean(nlpds);
end


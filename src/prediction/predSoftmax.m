function [pstar,pall] = predSoftmax(m,conf,xstar,ystar)
%PREDSOFTMAX [pstar,pall] = predSoftmax(m,conf,xstar,ystar)
%   Prediction for the multi-class classification model.
%
%   pstar(Y(x*_n) = y*_n) = \int p(y*_n | f*_n) p(f*_n|y,x*_n)
%      = E_q p(y*_n | f*_n),        q = p(f*_n | y,x*_n)
%      = 1/S \sum_s p(y*_n | f*_n,s),   f*_n,s ~ q
%   i.e. the predicted class probability is the average of squashing the
%   samples from the latent predictive distribtuion
%
%   pall(n,c) = p(Y(x*_n) = c) | x*_n) for all c 
%   Note that pall does not depend on the particular values of ystar.
%
% 14/05/14
nsamples = 2000;
[fmu,fvar] = predRegression(m,conf,xstar);
% each column is a sample of fstar (i.e. size N x C)
fss = mvnrnd(fmu', fvar', nsamples)';
ntest = size(ystar,1);
loglh = fastLikelihood(@llhSoftmax,ystar,fss,m.pars.hyp,ntest,m.Q);
loglh = reshape(loglh,ntest,nsamples);
pstar = mean(exp(loglh),2);
pall = [];
C = size(ystar,2);
for c=1:C
% p(Y(x*_n) = c) for all x*_n
yc = zeros(ntest,C);
yc(:,c) = 1;
loglh = fastLikelihood(@llhSoftmax,yc,fss,m.pars.hyp,ntest,m.Q);
loglh = reshape(loglh,ntest,nsamples);
pc = mean(exp(loglh),2);
pall = [pall; pc];
end
% now each row of pall, pall(n,:) is the probabilities of all classes
pall = reshape(pall,ntest,C);
end


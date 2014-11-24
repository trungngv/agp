function [meanval,varval,q5,q95] = lognormalMoments(mu,v)
%LOGNORMALMOMENTS [meanval,varval,q5,q95] = lognormalMoments(mu,v)
%   
%  Computes the mean and variance of a log-normal distribution.
%  Given f ~ N(mu,v) and y = exp(f) ~ ln N(mu,v), it returns
%     E[exp(f)] = exp(mu + v/2)
%     Var(exp(f)) = (exp(v)-1)*exp(2*mu+v)
meanval = exp(mu + v/2);
varval = (exp(v)-1).*exp(2*mu+v);
q5 = exp(mu-sqrt(v)*1.96);
q95 = exp(mu+sqrt(v)*1.96);
end


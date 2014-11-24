function logp = jointRegression(x,y,f,hyp)
%JOINTREGRESSION  logp = jointRegression(x,y,f,hyp)
%   
% The log joint of a standard GP regression model, i.e. the likelihood is 
% Gaussian with iid noise and the prior is also Gaussian.
sn2 = exp(2*hyp.lik);
logllh = -(0.5/sn2)*sum((y-f).^2)-log(2*pi*sn2)/2;
logprior = feval(@priorGaussian,f,x,hyp);
logp = logllh + logprior;
end


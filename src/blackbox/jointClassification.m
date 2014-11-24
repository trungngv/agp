function logp = jointClassification(x,y,f,hyp)
%JOINTCLASSIFICATION logp = jointClassification(x,y,f,hyp)
%   
% The log joint of a standard GP binary classification model, i.e. 
% - likelihood is probit/error function (likErf) or logistic (likLogistic)
% - the prior is Gaussian.
likfunc = @likLogistic;
logllh = feval(likfunc, [], y, f);
logprior = feval(@priorGaussian,f,x,hyp);
logp = sum(logllh) + logprior;
end


function logp = jointWarp(x,y,f,hyp)
%JOINTWARP logp = jointWarp(x,y,f,hyp)
%   
% The log joint of a warped model i.e. 
% - likelihood is a non-linear transformation and the prior is Gaussian.
logllh = llhWarp(y,f,hyp);
logprior = feval(@priorGaussian,f,x,hyp);
logp = sum(logllh) + logprior;
end


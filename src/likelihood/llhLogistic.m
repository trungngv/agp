function logllh = llhLogistic(y,f,hyp)
%LLHGAUSSIAN logllh = llhLogistic(y,f,hyp)
%   The standard logistic likelihood. We just use the GPML implementation
%   because both y and f are just vectors for this model.
% 
% All of the likelihood models should implement with this signature.
%   logllh = llhFunction(y,f,hyp) where
%      y : N x P vector of outputs (P > 1 for multi-output)
%      f : N x Q matrix of latent function values
%      hyp: any likelihood hyperparameters needed by the model
%      logllh: N x 1 with logllh(n) = log p(y_n | f_(n))
logllh = likLogistic([], y, f);
%logllh = likErf([], y, f);
end


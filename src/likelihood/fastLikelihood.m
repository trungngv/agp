function [logllh,dhyp] = fastLikelihood(likfunc,y,fs,hyp,N,Q)
%FASTLIKELIHOOD [logllh,dhyp] = fastLikelihood(y,fs,hyp,N,Q)
%   
% Vectorized computation of likelihood 'likfunc' for multiple sets of latent values
% e.g. samples. Each column in fs a sample of the latent values.
% 
% 03/05/14
nsamples = size(fs,2);
% vectorize computation for log llh
% cfs{i} = i-th sample with size (NxQ)x1 i.e. column vector
cfs = mat2cell(fs(:),repmat(N*Q,nsamples,1),1);
% cfs{i} = i-th sample with size NxQ, i.e. matrix
cfsmat = cellfun(@(A) reshape(A,N,Q),cfs,'UniformOutput',false);
fsmat = cell2mat(cfsmat); % size (N*S)xQ
if nargout == 1
  logllh = feval(likfunc, repmat(y,nsamples,1), fsmat, hyp);
else
  [logllh,dhyp] = feval(likfunc, repmat(y,nsamples,1), fsmat, hyp);
end


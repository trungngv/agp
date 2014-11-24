function logllh = llhSoftmax(y,f,hyp)
%LLHSOFTMAX logllh = llhSoftmax(y,f,hyp)
%   The (log) soft-max likelihood for multi-class classification.
%   
%   y_n : binary vector with only one element with value 1 indicating the
%         class
%   f_n : 1xC the vector of values of the C latent functions
%
if ~all(size(y) == size(f))
  error('y and f must have the same dimession')
end
% may use logsum here?
logllh = sum(y.*f,2) - log(sum(exp(f),2));
end


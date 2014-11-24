function errorrate = multiclassErrate(ystar,pall)
%MULTICLASSERRATE errorrate = multiclassErrate(ystar,pall)
%   
% Error rate for multi-class classification.
%
% ystar : the true classes
% pall : each row contains probabilities of an observation to be in the
%        classes
C = size(ystar,2);
labels = labelsFromBinary(ystar);
% prediction = the class with highest probability
preds = labelsFromBinary(pall == repmat(max(pall,[],2),1,C));
errorrate = sum(preds ~= labels)/numel(labels);
end


function labels = labelsFromBinary(y)
%LABELSFROMBINARY labels = labelsFromBinary(ystar)
%   
% Get the class lables from the binary matrices (one-of-K) representation.
[r,c] = find(y); % column with value 1 is the label
[~,sortInd] = sort(r);
labels = c(sortInd);
end


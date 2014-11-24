function yy = oneOfK(y, K)
% y(n) in {1,...,K}
% yy(n,:) is a K dimensional bit vector, where yy(n, y(n)) = 1
% Example
%>> oneOfK([1 2 1 3], 4)
%     1     0     0     0
%     0     1     0     0
%     1     0     0     0
%     0     0     1     0
     
if nargin < 2, K = length(unique(y)); end
%S = setdiff(1:K, unique(y));
%if ~isempty(S), error(sprintf('labels must be in {1,...,%d}', K)); end
if(~all(ismember(unique(y),1:K))),error('labels must be in {1,...,%d}', K); end
N = length(y);
yy = zeros(N, K);
ndx = sub2ind(size(yy), 1:N, y(:)');
yy(ndx) = 1;


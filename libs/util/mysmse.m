function [smse,sse] = mysmse(ytrue,ypred,trainmean)
%MYSMSE smse = mysmse(ytrue, ypred,trainmean)
%   Compute the standardised mean square error (SMSE), also NMSE in some
%   publications.
% 
% SSE = squared error / mean test variance 
% SMSE = mean(SSE)
% 12/05/14
sse = ((ypred-ytrue).^2)/mean((trainmean-ytrue).^2);
smse = mean(sse);
end


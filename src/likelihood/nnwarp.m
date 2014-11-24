function [t,dy] = nnwarp(y,ea,eb,c)
%NNWARP  [t,dy] = nnwarp(y,ea,eb,c)
%   Compute the transformation via neural-net warping function and its
%   derivative wrt the target / observed outputs y.
% 
%   t(y) = \sum_i=1 a_i tanh(b_i(y + c_i)), a_i,b_i >= 0
% 
t = y; % t(y)
dy = ones(size(y,1),1);
for i = 1:length(ea)
  tanhbyc = tanh(eb(i)*(y+c(i)));
  t = t + ea(i)*tanhbyc;
  dy = dy + ea(i)*(1-tanhbyc.^2)*eb(i);
end


function [logllh,dhyp] = llhWarp(y,f,hyp)
%LLHWARP [logllh,dhyp] = llhWarp(y,f,hyp)
%   
% The log likelihood for warped Gaussian processes and its derivatives.
%   p(y|f) = dt(y)/dy p(t(y)|f)
% where t(y) = nnwarp(y)
%
% The likelihood parameters are 
% hyp.lik = [a
%            b
%            c
%        log(sqrt(sn2))]
% where a,b,c are parameter vectors of the warping t(y).
%
% 30/04/14
num = (numel(hyp.lik)-1)/3;
dhyp = zeros(size(hyp.lik));
sn2 = exp(2*hyp.lik(end));
ea = zeros(num,1); eb = zeros(num,1); c = zeros(num,1);
for i = 1:num
  ea(i) = exp(hyp.lik(i));
  eb(i) = exp(hyp.lik(num+i));
  c(i) = hyp.lik(2*num+i);
end
% w = dt(y)/dy = 1 + \sum_i ea(i) eb(i) sech(s(i))^2
[t,w] = nnwarp(y, ea, eb, c);
tmphyp.lik = hyp.lik(end);
[logt,dhyp(end)] = llhGaussian(t,f,tmphyp);
logllh = log(w) + logt;

% using fixed warping parameters for now
% s = cell(num,1); sechs2 = cell(num,1); tanhs{i} = cell(num,1);
% for i = 1:num % useful constructs
%   s{i} = eb(i)*(y + c(i));
%   sechs2{i} = sech(s{i}).^2; % 1 - tanh(s)^2
%   tanhs{i} = tanh(s{i});
% end
% 
% for i = 1:num % derivatives wrt warping parameters
%   dhyp(i) = ea(i)*sum(eb(i)*sechs2{i}./w - (t-f).*tanhs{i}/sn2);
%   dhyp(num+i) = ea(i)*sum(sechs2{i}.*((eb(i)-2*eb(i)*s{i}.*tanhs{i})./w - (t-f).*s{i}/sn2));
%   dhyp(2*num+i) = ea(i)*eb(i)*sum(sechs2{i}.*(-2*eb(i)*tanhs{i}./w - (t-f)/sn2));
% end


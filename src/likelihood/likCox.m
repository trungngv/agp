function [lp,dlp] = likCox(hyp,y,f,dum1,dum2)
%LIKCOX [lp,dlp] = likCox(hyp,y,f,dum1,dum2)
%   The Poisson likelihood for the Log Gaussian Cox Process. 
%     p(y_n | f_n) = lambda_n^y_n exp(-lambda_n) / y_n!,
%   where lambda_n = exp(f_n + offset).
%     log p(y_n | f_n) = y_n log (lambda_n) - lambda_n - log(y_n!)
%   The function gammaln(y_n + 1) can be use in place of log(y_n!).
%
%   y : N x P vector of outputs (P > 1 for multi-output)
%   f : N x Q matrix of latent function values
hypc.offset = hyp(end);
lp = llhCoxGP(y,f,hypc);
dlp = y-exp(f+hyp(end)); % y-lambda
end


function [lp, ymu, ys2] = predWarp(m,conf,xstar,ystar)
%PREDWARP [lp, ymu, ys2] = predWarp(m,conf,xstar,ystar)
%   Prediction by the warped GPs model.
%     p(y*|x*) = dt(y*)/d(y*) p(t(y*)|x*)
%
% Adpated from the code by Edward Snelon.
%
% Note that the variance and log predictive returned is for the noisy prediction.
%
% INPUT
%   m : learned model
%   conf: model configurations
%   xstar : test inputs
%
% OUTPUT
%   lp = log predictive density (if ystar given)
%   ymu : preditive mean
%   ys2 : the approximate varianace computed from quantiles
% 
% 30/04/14
[muz, s2z] = predRegression(m,conf,xstar);
hyp = m.pars.hyp;
num = (numel(hyp.lik)-1)/3;
ea = zeros(num,1); eb = zeros(num,1); c = zeros(num,1);
for i = 1:num
  ea(i) = exp(hyp.lik(i));
  eb(i) = exp(hyp.lik(num+i));
  c(i) = hyp.lik(2*num+i);
end
s2z = s2z + exp(2*hyp.lik(end)); % need to use noisy prediction here
[N,D] = size(xstar);
% warp training targets to 'z' space
z = nnwarp(m.y, ea, eb, c);
[sortz,I] = sort(z); sortt = m.y(I);
alpha = [0.1:0.1:1-0.1]; % quantiles for predictive density
% find locations in latent 'z' space of quantiles specified by alpha
q = repmat(sqrt(2*s2z),1,length(alpha)).* ... 
	   repmat(erfinv(2*alpha-1),N,1) ...
	    + repmat(muz,1,length(alpha));
% pass quantiles through inverse warp function to find locations in
% observation 'y' space
quant = invert(q, ea, eb, c, sortz, sortt);
ys2 = ((quant(:,9)-quant(:,1))/4).^2;

% quadrature to compute mean of predictive density
quadr = [0.3429013 1.0366108 1.7566836 2.5327317 3.4361591];
quadr = [-quadr(end:-1:1) quadr];
H = [0.6108626 0.2401386 0.0338744 0.0013436 .00000076];
H = [H(end:-1:1) H];

ymu = invert(sqrt(2*s2z)*quadr + ... 
	     repmat(muz,1,length(quadr)),ea,eb,c,sortz,sortt);
ymu = ymu*H'/sqrt(pi);

% if test targets are supplied, compute log predictive density
if nargin == 4
  [tstar,w] = nnwarp(ystar,ea,eb,c);
  lp = -0.5*log(2*pi*s2z) -0.5*(tstar-muz).^2./s2z + log(w);
else
  lp = [];
end
end

function newt = invert(newz, ea, eb, c, sortz, sortt)
% invert warp function for vector newz. sortz and sortt provide
% very good initial starting points for Newton iterations 

for j = 1:size(newz,1)
  for k = 1:size(newz,2)
    if newz(j,k) > sortz(end)
      t0(j,k) = sortt(end);
    elseif newz(j,k) < sortz(1)
      t0(j,k) = sortt(1);
    else
      I = find(sortz > newz(j,k)); I = [I(1)-1;I(1)];
      t0(j,k) = mean(sortt(I));
    end
  end
end

newt = warpinv(newz,ea,eb,c,t0,8); % may need to adjust no. of
                                   % iterations to ensure convergence

end


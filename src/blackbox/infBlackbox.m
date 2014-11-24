function m = infBlackbox(m,conf)
%INFBLACKBOX m = infBlackbox(m,conf)
%   
% Blackbox variational inference for models with a standard GP latent function.
rho = conf.rho;
t = conf.temperature;
N = size(m.x,1);

% repeat until convergence
iter = 1;
Sigma = m.pars.L*m.pars.L';
pSamples = 200; % for control variates
% cumulative grad2 of variational parameters for adaptive learn rate
cumdM2 = zeros(N,1);
cumdL2 = zeros(N,N);
cumdhyp2 = zeros(size(m.pars.hyp.cov));
fval = zeros(conf.maxiter,1);
while true
  %-----------------------------------------------------
  % Monte Carlo approximation of the gradient of variational parameters
  % generate samples
  fs = mvnrnd(m.pars.M', Sigma, conf.nsamples);
  [fval(iter),dM,dL] = computeNoisyGradient(m,fs,pSamples,conf);
  % stochastic update with the noisy gradients
  if conf.useAdagrad
    cumdM2 = cumdM2 + dM.^2 + 1e-10;
    cumdL2 = cumdL2 + dL.^2 + 1e-10;
    delta_m = (rho.mu ./ sqrt(cumdM2)).*dM;
    delta_l = (rho.lambda ./ sqrt(cumdL2)).*dL;
  else
    delta_m = lrate(rho.mu,iter,t.mu)*dM;
    delta_l = lrate(rho.lambda,iter,t.lambda)*dL;
  end
  delta = norm(delta_m) + norm(delta_l);
  m.pars.M = m.pars.M + delta_m;
  m.pars.L = m.pars.L + delta_l;
  Sigma = m.pars.L*m.pars.L';
  fprintf('variational change M= %.4f\n', norm(delta_m));
  fprintf('variational change L= %.4f\n', norm(delta_l));
  
  %------------------------------------------------------------
  % Monte Carlo approximation of the covariance hyperparameters (M-step)
  if iter > 20
    fs = mvnrnd(m.pars.M', Sigma, conf.nsamples);
    grad = zeros(numel(m.pars.hyp.cov),1);
    grad2 = grad;
    for i=1:conf.nsamples
      fi = fs(i,:)';
      [~,thisgrad] = feval(m.pdist, fi, m.x, m.pars.hyp);
      grad = grad + thisgrad;
      grad2 = grad2 + thisgrad.^2;
    end
    grad = grad / conf.nsamples;
    grad2 = grad2 / conf.nsamples;
    vargrad = grad2 - grad.^2;
    disp('hyp variance: ')
    disp(vargrad')
  
    % stochastic update with the noisy gradients
    cumdhyp2 = cumdhyp2 + grad.^2;
    if conf.useAdagrad
      delta_hyp = (rho.hyp ./ sqrt(cumdhyp2)) .* grad;
    else
      delta_hyp = lrate(rho.hyp,iter,t.hyp)*grad;
    end
    m.pars.hyp.cov = m.pars.hyp.cov + delta_hyp;
    fprintf('hyp change = %.4f\n', norm(delta_hyp));
  end

  if mod(iter,conf.displayInterval) == 0 || iter == 1
    if (strcmp(func2str(m.pred),'predRegression'))
      [fmu,fvar] = feval(m.pred, m,conf,m.x);
      figure; hold on;
      plotMeanAndStd(m.x,fmu,2*sqrt(fvar),[7 7 7]/8);
      fprintf('predictive likelihood: %.4f\n', -mynlpd(m.y,fmu,fvar));
    elseif strcmp(func2str(m.pred),'predClassification')
      % temp code for classification
      figure; hold on;
      [t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
      tt = [t1(:) t2(:)];
      lp = feval(m.pred, m, conf, tt);
      contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
      lp = feval(m.pred, m, conf, m.x);
      ind1 = exp(lp) > 0.5;
      ind2 = ~ind1;
      scatter(m.x(ind1,1),m.x(ind1,2),'x');
      scatter(m.x(ind2,1),m.x(ind2,2),'o');
    elseif strcmp(func2str(m.pred),'predWarp')
      [~,fmu,fvar] = feval(m.pred, m, conf, m.xt);
      figure; hold on;
      plot(m.xt,m.yt,'b.'); % true data
      plot(m.xt,fmu,'r.');
      plot(m.xt,fmu+2*sqrt(fvar),'r-');
      plot(m.xt,fmu-2*sqrt(fvar),'r-');
      xlim([-pi pi])
      ylim([-1.5 1.5])
    end
    
  end
  
  iter = iter + 1;
  if delta < 1e-2 || iter > conf.maxiter 
    break
  end
end

figure; hold on;
plot(fval);
title('objective function');

end

function [fval,dM,dL] = computeNoisyGradient(m,fs,pSamples,conf)
nvars = numel(m.pars.M)+numel(m.pars.L);
N = size(m.x,1);
% estimate the optimal control variates factor 
pilot_y = zeros(pSamples, nvars);
pilot_z = zeros(pSamples, nvars);
for i=1:pSamples
  fi = fs(i,:)';
  [dM,dL,logq] = feval(m.vdist, fi, m.pars.M, m.pars.L);
  logp = feval(m.jdist, m.x, m.y, fi, m.pars.hyp);
  pilot_z(i,:) = [dM(:); dL(:)];
  thisgrad = (logp-logq)*pilot_z(i,:);
  pilot_y(i,:) = thisgrad;
end
above = sum((pilot_y - repmat(mean(pilot_y),pSamples,1)).*pilot_z)/(pSamples-1);
below = sum(pilot_z.^2)/(pSamples-1);
aopt = above ./ below;
aopt(isnan(aopt)) = 0;

% compute the noisy gradient using the control variates
grad = zeros(nvars,1);
grad2 = grad;
ugrad = grad; ugrad2 = grad; % gradienst without control variates
fval = 0;
for i=1:conf.nsamples
  fi = fs(i,:)';
  [dM,dL,logq] = feval(m.vdist, fi, m.pars.M, m.pars.L);
  controller = [dM(:); dL(:)];
  logp = feval(m.jdist, m.x, m.y, fi, m.pars.hyp);
  fval = fval + logp - logq;
  thisgrad = (logp - logq)*controller;
  grad = grad + thisgrad - (aopt').*controller;
  if conf.checkVarianceReduction
    grad2 = grad2 + (thisgrad - (aopt').*controller).^2;
    ugrad = grad + thisgrad;
    ugrad2 = ugrad2 + thisgrad.^2;
  end
end
grad = grad / conf.nsamples;
% check variance reduction
if conf.checkVarianceReduction
  grad2 = grad2 / conf.nsamples;
  ugrad = ugrad / conf.nsamples;
  ugrad2 = ugrad2 / conf.nsamples;
  vargrad = grad2 - grad.^2;
  varugrad = ugrad2 - ugrad.^2;
  disp('diff in estimated grad')
  disp(nanmean(abs(grad-ugrad)))
  reduction = 100*(varugrad - vargrad)./varugrad; 
  disp('min, max, mean percentage of variance reduction:')
  disp([nanmin(reduction), nanmax(reduction), nanmean(reduction)])
  disp('min, max, mean controlled variance:')
  disp([min(vargrad), max(vargrad), mean(vargrad)])
end

dM = grad(1:N);
dL = reshape(grad(N+1:end),N,N);
fval = fval / conf.nsamples;
end
%%  code that estimate control variates while estmating gradients
%   allgrad = zeros(conf.nsamples,10);
%   for i=1:conf.nsamples
%     fi = fs(i,:)';
%     [dmu,dLambda,logq] = feval(m.vdist, fi, m.pars.M, m.pars.L);
%     thisgrad = (logp - logq)*[dmu(:); dLambda(:)];
%     logp = feval(m.jdist, m.x, m.y, fi, m.pars.hyp);
%     if (i <= pilotSamples)
%       pilot_y(i,:) = thisgrad';
%       pilot_z(i,:) = [dmu(:); dLambda(:)];
%     end
%     grad = grad + thisgrad;
%     controller = controller + [dmu(:); dLambda(:)]; % the function h
%     allgrad(i,:) = thisgrad(1:10)';
%   end
%   grad = grad / conf.nsamples;
%   
%   % control variates
%   above = sum((pilot_y - repmat(mean(pilot_y),pilotSamples,1)).*pilot_z)/(pilotSamples-1);
%   below = sum(pilot_z.^2)/(pilotSamples-1);
%   a = above ./ below;
%   grad = grad - a.*controller/conf.nsamples;

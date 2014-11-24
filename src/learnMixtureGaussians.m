function m = learnMixtureGaussians(m,conf)
%LEARNMIXTUREGAUSSIANS m = learnMixtureGaussians(m,conf)
%   
% Variational inference with mixture of Gaussians posterior for GP models.
% This code uses LBFGS or CG to optimize the variational parameters instead
% of stochastic optimization.
%
% 26/05/14
Q = m.Q;
N = m.N;
iter = 1;
fval = [];
conf.cvSamples = 200;
if ~isfield(conf,'learnhyp')
  conf.learnhyp = true;
end
if ~isfield(conf,'latentnoise')
  sn2 = 0;
else
  sn2 = conf.latentnoise;
end
LKchol = cell(Q,1);
while true
  % E-step : optimize variational parameters
  if iter == 1
    for j=1:Q
      K = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.x) + sn2*eye(N);
      LKchol{j} = jit_chol(K);
    end
  end
  theta = [m.pars.M(:); m.pars.L(:)];
  opts = struct('Display','final','Method','lbfgs','MaxIter',10,...
    'MaxFunEvals',100,'DerivativeCheck','off'); 
  [theta,fX,~] = minFunc(@elbo,theta,opts,m,conf,LKchol);
  %[theta,fX,~] = minimize(theta,@elbo,10,m,conf,K,LKchol);
  fval = [fval; fX(end)];
  delta_m = mean(mean(abs(m.pars.M(:)-theta(1:numel(m.pars.M)))));
  delta_l = mean(mean(abs(m.pars.L(:)-theta(numel(m.pars.M)+1:end))));
  fprintf('variational change m= %.4f\n', delta_m);
  fprintf('variational change s= %.4f\n', delta_l);
  m.pars.M = reshape(theta(1:numel(m.pars.M)),size(m.pars.M));
  m.pars.L = reshape(theta(numel(m.pars.M)+1:end),size(m.pars.L));
  if (delta_m + delta_l)/2 < 1e-3 || (iter > 1 && fval(end-1) - fX(end) < 1e-5)
    break;
  end
  
  if conf.learnhyp
    %--- gradient-based optimization for covariance hyperparameters
  % m.pars.hyp.cov = minimize(m.pars.hyp.cov, @elboCovhyp, 5, m, sn2);
    opts = struct('Display','final','Method','lbfgs','MaxIter',5,...
        'MaxFunEvals',100,'DerivativeCheck','off'); 
    hyp0 = minFunc(@elboCovhyp,unwrap(m.pars.hyp.cov),opts,m,sn2);
    m.pars.hyp.cov = rewrap(m.pars.hyp.cov,hyp0);
    for j=1:Q
      K = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.x) + sn2*eye(N);
      LKchol{j} = jit_chol(K);
    end
    fval = [fval; elbo(theta,m,conf,LKchol)];
  end
  
  %--- update likelihood parameters
  if (numel(m.pars.hyp.lik) > 0)
    fs = cell(m.K,1);
    for k=1:m.K
      Sk = exp(2*m.pars.L(:,k));
      % generate S samples f ~ \Normal(f; m_k, S_k)
      fs{k} = mvnrnd(m.pars.M(:,k)', Sk', conf.nsamples)';
    end
    opts = struct('Display','final','Method','lbfgs','MaxIter',5,...
      'MaxFunEvals',100,'DerivativeCheck','off'); 
    lik0 = minFunc(@elbolik,m.pars.hyp.lik(end),opts,m,fs);
%    lik0 = minimize(m.pars.hyp.lik(end),@elbolik,5,m,fs);
    m.pars.hyp.lik(end) = lik0;
    fval = [fval; elbo(theta,m,conf,LKchol)];
    disp('new lik hyp = ')
    disp(exp(2*m.pars.hyp.lik(end)));
  end
  
  iter = iter + 1;
  if iter > conf.maxiter %|| delta < 1e-2
    break
  end
end

fval = -fval(fval ~= 0);
figure; hold off;
plot(1:numel(fval),fval,'-');
title('evidence lower bound');
m.fval = fval;

end

% the negative elbo and its gradient wrt variational parameters
function [fval,grad] = elbo(theta,m,conf,LKchol)
  % fix the random number geneator to make sure that same value of theta
  % gives the same function value and gradient
  rng(10101,'twister');
  m.pars.M = reshape(theta(1:numel(m.pars.M)),size(m.pars.M));
  m.pars.L = reshape(theta(numel(m.pars.M)+1:end),size(m.pars.L));
  dM = zeros(size(m.pars.M));
  dL = zeros(size(m.pars.L));
  w = exp(m.pars.w);
  S = exp(2*m.pars.L);

  % entropy part
  fval1 = 0;
  logNkl = zeros(m.K,m.K);
  logq = zeros(m.K,1);
  for k=1:m.K
    for l=1:m.K
      logNkl(k,l) = varDiagGaussian(m.pars.M(:,k),m.pars.M(:,l),log(sqrt(S(:,k)+S(:,l))));
    end
    logq(k) = logsum(log(w) + logNkl(k,:)');
    fval1 = fval1 - w(k)*logq(k);
  end
  % neg cross entropy part
  sumlogdet = 0;
  Kinv = cell(m.Q,1);
  for j=1:m.Q
    Kinv{j} = invChol(LKchol{j});
    sumlogdet = sumlogdet + 2*sum(log(diag(LKchol{j})));
  end
  fval2 = repmat(m.Q*m.N*log(2*pi) + sumlogdet,m.K,1);
  
  % derivatives
  for k=1:m.K
    % entropy part
    for l=1:m.K
      common = w(l)*(exp(logNkl(k,l)-logq(k))+exp(logNkl(k,l)-logq(l)));
      Skl = S(:,k)+S(:,l);
      Mkl = m.pars.M(:,k)-m.pars.M(:,l);
      dM(:,k) = dM(:,k) + w(k)*common*Mkl./Skl;
      dL(:,k) = dL(:,k) + w(k)*common*(1./Skl - (Mkl./Skl).^2).*S(:,k);
    end
    % cross entropy part
    for j=1:m.Q
      s_row = (j-1)*m.N+1; e_row = j*m.N;
      Kinvm = solve_chol(LKchol{j},m.pars.M(s_row:e_row,k));
      Skj = S(s_row:e_row,k);
      fval2(k) = fval2(k) + m.pars.M(s_row:e_row,k)'*Kinvm + sum(diag(Kinv{j}).*Skj);
      dM(s_row:e_row,k) = dM(s_row:e_row,k)-w(k)*Kinvm;
      dL(s_row:e_row,k) = dL(s_row:e_row,k)-w(k)*diag(Kinv{j}).*Skj;
    end
  end
  fval2 = -0.5*sum(w.*fval2);
  
  % compute noisy graidents for each mixture component
  fvalk = zeros(m.K,1);
  for k=1:m.K
    Sk = exp(2*m.pars.L(:,k));
    % generate S samples f ~ \Normal(f; m_k, S_k)
    fs = mvnrnd(m.pars.M(:,k)', Sk', conf.nsamples)';
    if nargout == 1
      fvalk(k) = computeNoisyGradient(m,fs,m.pars.M(:,k),m.pars.L(:,k),conf);
    else
      [fvalk(k),dM1,dL1] = computeNoisyGradient(m,fs,m.pars.M(:,k),m.pars.L(:,k),conf);
      dM(:,k) = dM(:,k) + exp(m.pars.w(k))*dM1;
      dL(:,k) = dL(:,k) + exp(m.pars.w(k))*dL1;
      grad = -[dM(:); dL(:)];
    end
  end
  fval = -(fval1 + fval2 + sum(fvalk));
  
end

% compute the noisy gradient using the given samples
function [fval,dM,dL] = computeNoisyGradient(m,fs,Mk,Lk,conf)
% m : model
% fs : the samples f ~ q(f | lambda_k)
  N = m.N; Q = m.Q;
  nsamples = size(fs,2);
  cvsamples = conf.cvSamples; % samples to estimate control variate factors
  logllh = fastLikelihood(m.likfunc,m.y,fs,m.pars.hyp,N,Q);
  fval = mean(logllh)*N; % E log p(y|f)
  if nargout > 1
    sinv = 1./exp(2*Lk);
    f0 = fs(:) - repmat(Mk,nsamples,1);
    dM1 = f0.*repmat(sinv,nsamples,1);
    dL1 = -1 + f0.*dM1;

    % some useful constructs
    logllh = reshape(logllh,N,nsamples);
    logllh = repmat(logllh,2*Q,1); % size (2*N*Q)xS
    dML = [reshape(dM1,N*Q,nsamples); reshape(dL1,N*Q,nsamples)]; % size (2*N*Q)xS

    % estimate the optimal control variates factor 
    pz = dML(:,1:cvsamples)';
    py = logllh(:,1:cvsamples)'.*pz;
    above = sum((py-repmat(mean(py),cvsamples,1)).*pz)/(cvsamples-1);
    below = sum(pz.^2)/(cvsamples-1); % var(z) with E(z) = 0
    cvopt = above ./ below;
    cvopt(isnan(cvopt)) = 0;

    % the noisy gradient using the control variates
    grads = logllh.*dML - repmat(cvopt',1,nsamples).*dML;
    grad = mean(grads,2);
    if conf.checkVarianceReduction
      ugrads = logllh.*dML;
      ugrad = mean(ugrads,2);
      vargrad = var(grads,0,2);
      varugrad = var(ugrads,0,2);
      disp('diff in estimated grad')
      disp(mean(abs(grad-ugrad)))
      reduction = 100*(varugrad - vargrad)./varugrad; 
      disp('min, max, mean percentage of variance reduction:')
      disp([min(reduction), max(reduction), mean(reduction)])
      disp('min, max, mean controlled variance:')
      disp([min(vargrad), max(vargrad), mean(vargrad)])
    end

    dM = grad(1:N*Q);
    dL = grad(N*Q+1:end);
  end
end

% elbo as a function of covariance hyperparameters
function [fhyp,dhyp] = elboCovhyp(hyp, m, sn2)
if iscell(hyp)
  m.pars.hyp.cov = hyp;
else
  m.pars.hyp.cov = rewrap(m.pars.hyp.cov,hyp);
end
Q = m.Q; N = m.N;
w = exp(m.pars.w); S = exp(2*m.pars.L);
Lchol = cell(Q,1); Kinv = cell(Q,1); sumlogdet = 0; dhyp = cell(Q,1);
for j=1:Q
  K = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.x) + sn2*eye(N);
  Lchol{j} = jit_chol(K);
  Kinv{j} = invChol(Lchol{j});
  sumlogdet = sumlogdet + 2*sum(log(diag(Lchol{j})));
  dhyp{j} = zeros(numel(m.pars.hyp.cov{j}),1);
end
fval = repmat(Q*N*log(2*pi) + sumlogdet,m.K,1);

% assuming all covfuncs take the same number of hyperparameters
dK = cell(Q,numel(m.pars.hyp.cov{1}));
for k=1:m.K
  for j=1:Q
    s_row = (j-1)*N+1; e_row = j*N;
    Kinvm = solve_chol(Lchol{j},m.pars.M(s_row:e_row,k));
    Skj = S(s_row:e_row,k);
    fval(k) = fval(k) + m.pars.M(s_row:e_row,k)'*Kinvm + sum(diag(Kinv{j}).*Skj);
    for i=1:numel(m.pars.hyp.cov{j})
      if isempty(dK{j,i})
        dK{j,i} = feval(m.pars.hyp.covfunc, m.pars.hyp.cov{j}, m.x, [], i);
      end
      dthetai = 0.5*w(k)*trAB(Kinvm*Kinvm' - Kinv{j} + Kinv{j}*diagAB(Skj,Kinv{j}), dK{j,i});
      dhyp{j}(i) = dhyp{j}(i) + dthetai;
    end
    if strcmp(func2str(m.pars.hyp.covfunc),'covSEard') && sn2 == 0
      dhyp{j}(i) = dhyp{j}(i) - dthetai + w(k)*(-N + m.pars.M(s_row:e_row,k)'*Kinvm...
        + sum(diag(Kinv{j}).*Skj));
    end
  end
end
fval = -0.5*sum(w.*fval);

fhyp = -fval;
if iscell(hyp)
  dhyp = rewrap(m.pars.hyp.cov,-unwrap(dhyp));
else
  dhyp = -unwrap(dhyp);
end
end

% elbo as a function of likelihood parameters
function [fval,dlikhyp] = elbolik(hyp,m,fs)
m.pars.hyp.lik(end) = hyp;
nsamples = size(fs,2);
fval = 0;
dlikhyp = 0;
for k=1:m.K
  [logllh,dlikthis] = fastLikelihood(m.likfunc,m.y,fs{k},m.pars.hyp,m.N,m.Q);
  dlikhyp = dlikhyp + dlikthis(end)/nsamples;
  fval = fval + sum(logllh)/nsamples;
end
dlikhyp = -dlikhyp/m.K;
fval = -fval/m.K;
end

rng(1110, 'twister');
clear functions
clear

DATA = 7;
[x,y,xt,yt,model] = getDataAndModel(DATA);

%-----------------------------------------------------------
[N,D] = size(x);
m.x = x; m.y = y; m.xt = xt; m.yt = yt;

m.pars.hyp.covfunc = 'covSEard';
m.pars.hyp.lik = log(sqrt(0.01));
%m.pars.hyp.lik = [];
m.pars.hyp.cov = log([ones(D,1); 1]);
m.pars.M = zeros(N,1);
m.pars.L = diag(ones(N,1));

conf.nsamples = 2000;
conf.covfunc = 'covSEard';
conf.maxiter = 100;
rho.mu = 1e-3;
rho.lambda = 1e-5;
rho.hyp = 0;
t.mu = 0;
t.lambda = 0;
t.hyp = 0.5;
conf.rho = rho;
conf.temperature = t;
conf.displayInterval = 5;
conf.checkVarianceReduction = true;
conf.useAdagrad = false;

if strcmp(model,'regression')
  m.pdist = @priorGaussian; % prior model
  m.jdist = @jointRegression; % joint model
  m.vdist = @varGaussian; % variation model
  m.pred = @predRegression; % prediction distribution
elseif strcmp(model,'classification')
  m.pdist = @priorGaussian;
  m.jdist = @jointClassification;
  m.vdist = @varGaussian;
  m.pred = @predClassification;
elseif strcmp(model,'warp')
  m.pdist = @priorGaussian;
  m.jdist = @jointWarp;
  m.vdist = @varGaussian;
  m.pred = @predWarp;
  m.pars.hyp.warp.ea = [0.6461    0.4252];
  m.pars.hyp.warp.eb = [8.4684  304.4854];
  m.pars.hyp.warp.c =  [0.0216   -0.0001];
end
tic;
m = infBlackbox(m,conf);
toc

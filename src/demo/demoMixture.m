%% See also demoFull.m for documentation of the parameters

clear; clc; close all
rng(1110, 'twister');

x = linspace(-5,5,100)';
y = sample_gp(x,'covSEard',[0.01,1],1) + 0.1*randn(size(x));
[N,D] = size(x);
Q = 1; % no. latent functions
K = 1; % no. mixture components

% data and parameters
m.x = x; m.y = y; m.N = N; m.Q = Q; m.K = K;
m.pars.M = zeros(N*Q,K);
m.pars.L = log(ones(N*Q,K));

% covariance hyperparameters
m.pars.hyp.covfunc = @covSEard;
m.pars.hyp.cov = cell(Q,1);
m.pars.hyp.cov{1} = log([ones(D,1); 1]);
m.pars.w = log(1/K)*ones(K,1);

% configurations
conf.nsamples = 2000;
conf.covfunc = @covSEard;
conf.maxiter = 100;
conf.displayInterval = 10;
conf.checkVarianceReduction = false;
conf.latentnoise = 0;

m.likfunc = @llhGaussian;
m.pars.hyp.likfunc = m.likfunc;
m.pred = @mixturePredRegression;
m.pars.hyp.lik = log(sqrt(0.01));

tic;
m = learnMixtureGaussians(m,conf);
toc

[fmu,~,yvar] = feval(m.pred, m, conf, m.x);
figure; hold on;
plotMeanAndStd(m.x,fmu,2*sqrt(yvar),[7 7 7]/8);
plot(m.x, m.y, 'o')


clear; clc; close all
rng(1110, 'twister');

%% Generate toy dataset
x = linspace(-5,5,100)';
y = sample_gp(x,'covSEard',[0.01,1],1) + 0.1*randn(size(x));
[N,D] = size(x);
Q = 1; % no. latent functions

% Data
m.x = x; m.y = y; m.N = N; m.Q = Q;

%% Parameter initialization
% variational parameters
m.pars.M                      = zeros(N*Q,1);     % the mean parameters
m.pars.L                      = zeros(Q*N,1);     % the linear parametrisation of the cov matrix
for j=1:Q
  % free initial values 
  m.pars.S{j} = diag(ones(N,1));                  % the cov matrix
end

% covariance hyperparameters
m.pars.hyp.covfunc = @covSEard;                   % cov function
m.pars.hyp.cov = cell(Q,1);                       % cov hyperparameters
m.pars.hyp.cov{1} = log([ones(D,1); 1]);

%% Optimization settings
conf.nsamples                 = 2000;             % number of samples for gradient estimator
conf.covfunc                  = @covSEard;        % covariance function
conf.maxiter                  = 100;              % how many optimization iterations to run?
conf.variter                  = 10;               % maxiter for optim the variational hyperparameter (per iteration)
conf.hypiter                  = 5;                % maxiter for optim the cov hyperparameter (per iteration)
conf.likiter                  = 5;                % maxiter for optim the likelihood hyperparameter (per iteration)
conf.displayInterval          = 20;               % intervals to display some progress 
conf.checkVarianceReduction   = false;            % show diagnostic for variance reduction?
conf.learnhyp                 = true;             
conf.latentnoise              = 0;                % minimum noise level of the latent function

%% Model setting
m.likfunc                     = @llhGaussian;     % likelihood function
m.pars.hyp.likfunc            = m.likfunc;      
m.pred                        = @predRegression;  % prediction 
m.pars.hyp.lik = log(sqrt(0.01));                 % likelihood parameters

tic;
m = learnFullGaussian(m,conf);
toc

%% Plot
[fmu,fvar,yvar] = feval(m.pred, m, conf, m.x);
figure; hold on;
plotMeanAndStd(m.x,fmu,2*sqrt(fvar),[7 7 7]/8);
plot(m.x, m.y, 'o')


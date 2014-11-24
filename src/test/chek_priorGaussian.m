function chek_priorGaussian()
%CHEK_PRIORGAUSSIAN chek_priorGaussian()
%   Gradient check for priorGaussian().
clear functions
n = 20; dim = 2;
x = linspace(-10,10,n)';
x = [x, x];
x = x + 0.01*rand(size(x));
f = sin(x(:,1)) + sin(x(:,2));
theta = log([ones(dim,1); 5]);
[mygrad, delta] = gradchek(theta', @fval, @gradval, f, x);
disp('maxdiff = ')
disp(max(abs(delta)))
end

function val = fval(theta, f, x)
hyp.covfunc = 'covSEard';
hyp.cov = theta';
val = priorGaussian(f,x,hyp);
end

function d = gradval(theta, f, x)
hyp.covfunc = 'covSEard';
hyp.cov = theta';
[~,d] = priorGaussian(f,x,hyp);
d = d';
end
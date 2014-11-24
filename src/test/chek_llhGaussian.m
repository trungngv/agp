function chek_llhGaussian()
%CHEK_LLHGAUSSIAN chek_llhGaussian()
%   Gradient check for llhGaussian().
clear functions
N = 100;
f = rand(N,1);
y = f + 0.1*rand(N,1);
theta = log(sqrt(0.1+rand));
[mygrad, delta] = gradchek(theta, @func, @grad, y, f);
disp('maxdiff = ')
disp(max(abs(delta)))
end

function val = func(theta, y, f)
hyp.lik = theta';
val = sum(llhGaussian(y,f,hyp));
end

function dhyp = grad(theta, y, f)
hyp.lik = theta';
[~,dhyp] = llhGaussian(y,f,hyp);
end
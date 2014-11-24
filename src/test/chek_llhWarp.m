function chek_llhWarp()
%CHEK_LLHWARP chek_llhWarp()
%   Gradient check for llhWarp().
N = 100;
f = rand(N,1);
y = f + 0.1*rand(N,1);
num = 2; % for warp
a = log(rand(num,1));
b = log(rand(num,1));
c = 0.5 - rand(num,1);
theta = zeros(3*num+1,1);
for i=1:num
  theta(i) = a(i);
  theta(num+i) = b(i);
  theta(2*num+i) = c(i);
end
theta(end) = log(sqrt(0.1+rand));
[mygrad, delta] = gradchek(theta', @func, @grad, y, f);
disp('max diff = ')
disp(max(abs(delta)))
end

function val = func(theta, y, f)
hyp.lik = theta';
val = sum(llhWarp(y,f,hyp));
end

function dhyp = grad(theta, y, f)
hyp.lik = theta';
[~,dhyp] = llhWarp(y,f,hyp);
dhyp = dhyp';
end
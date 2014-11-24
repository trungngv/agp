function chek_lowerboundEntropy()
%CHEK_LOWERBOUNDENTROPY chek_lowerboundEntropy()
%   Gradient check for lowerboundEntropy().
Q = 5;
N = 50;
K = 3;
epsilon = 1e-10;
M0 = rand(Q*N,K); % (Q*N)*K
S0 = log(1+rand(Q*N,K));
w0 = log(rand(K,1)+epsilon);
theta = [M0(:); S0(:); w0];
[mygrad, delta] = gradchek(theta', @f, @grad);
disp('maxdiff = ')
disp(max(abs(delta)))

  function val = f(theta)
    d = Q*N;
    M = reshape(theta(1:K*d),d,K);
    L = reshape(theta(K*d+1:2*K*d),d,K);
    w = theta(2*K*d+1:end)';
    val = lowerboundEntropy(M,L,w);
  end

  function d = grad(theta)
    d = Q*N;
    M = reshape(theta(1:K*d),d,K);
    L = reshape(theta(K*d+1:2*K*d),d,K);
    w = theta(2*K*d+1:end)';
    [~,dM,dS,dw] = lowerboundEntropy(M,L,w);
    d = [dM(:);dS(:);dw]';
  end
end


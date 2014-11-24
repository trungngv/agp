function chek_negCrossEntropy()
%CHEK_NEGCROSSENTROPY chek_negCrossEntropy()
%   Gradient check for negCrossEntropy().
rng(1110,'twister');
D = 3;
Q = 3;
N = 100;
K = 3;
epsilon = 1e-10;
M0 = rand(Q*N,K); % (Q*N)*K
L0 = log(1+rand(Q*N,K));
w0 = log(rand(K,1)+epsilon);
x = 5*rand(N,D);
hyp.covfunc = @covSEard;
for j=1:Q
  hyp.cov{j} = log([ones(D,1); 2] + 1e-2*rand(D+1,1));
end

% gradient checks for hyperparameters
theta = unwrap(hyp.cov);
[mygrad, delta] = gradchek(theta', @fhyp, @gradhyp, M0, L0, w0, hyp, x);
disp('maxdiff = ')
disp(max(abs(delta)))
disp('press any key')
pause

% gradient check for variational parameters
theta = [M0(:); L0(:); w0];
[mygrad, delta] = gradchek(theta', @f, @grad, hyp, x);
disp('maxdiff = ')
disp(max(abs(delta)))

  function val = f(theta, hyp, x)
    d = Q*N;
    M = reshape(theta(1:K*d),d,K);
    L = reshape(theta(K*d+1:2*K*d),d,K);
    w = theta(2*K*d+1:end)';
    val = negCrossEntropy(M,L,w,hyp,x);
  end

  function val = fhyp(theta, M, L, w, hyp, x)
    hyp.cov = rewrap(hyp.cov,theta');
    val = negCrossEntropy(M,L,w,hyp,x);
  end

  function d = grad(theta, hyp, x)
    d = Q*N;
    M = reshape(theta(1:K*d),d,K);
    L = reshape(theta(K*d+1:2*K*d),d,K);
    w = theta(2*K*d+1:end)';
    [~,dM,dL,dw] = negCrossEntropy(M,L,w,hyp,x);
    d = [dM(:);dL(:);dw]';
  end

  function d = gradhyp(theta, M, L, w, hyp, x)
    hyp.cov = rewrap(hyp.cov,theta');
    [~,~,~,~,dcell] = negCrossEntropy(M,L,w,hyp,x);
    d = unwrap(dcell)';
  end
end


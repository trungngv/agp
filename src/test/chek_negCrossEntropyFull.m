function chek_negCrossEntropyFull()
%CHEK_NEGCROSSENTROPYFULL chek_negCrossEntropyFull()
%   Gradient check for negCrossEntropyFull().
rng(1110,'twister');
D = 3;
Q = 3;
N = 100;
x = 5*rand(N,D);
hyp.covfunc = @covSEard;
for j=1:Q
  hyp.cov{j} = log([ones(D,1); 2] + 1e-2*rand(D+1,1));
end
M0 = rand(Q*N,1); % (Q*N)*K
L0 = [];
for j=1:Q
  A = rand(N,N);
  Sigma = A*A';
  Lj = jit_chol(Sigma)';
  L0 = [L0; Lj];
end

% gradient checks for hyperparameters
theta = [];
for j=1:Q
  theta = [theta; hyp.cov{j}];
end
[mygrad, delta] = gradchek(theta', @fhyp, @gradhyp, M0, L0, hyp, x);
disp('maxdiff = ')
disp(max(abs(delta)))
disp('press any key')
pause

%gradient check for variational parameters
% theta = [M0(:); L0(:)];
% [mygrad, delta] = gradchek(theta', @f, @grad, hyp, x);
% disp('maxdiff = ')
% disp(max(abs(delta)))
% disp('press any key')

  function val = f(theta, hyp, x)
    M = theta(1:Q*N)';
    % only the lower triangular are variables
    L = reshape(theta(Q*N+1:end),Q*N,N);
    trueL = L;
    for jj=1:Q
      start_row = (jj-1)*N+1;
      end_row = jj*N;
      Lj = tril(L(start_row:end_row,:));
      trueL(start_row:end_row,:) = Lj;
    end
    val = negCrossEntropyFull(M,trueL,hyp,x);
  end

  function val = fhyp(theta, M, L, hyp, x)
    pos = 1;
    for jj=1:Q
      hyp.cov{jj} = theta(pos:pos+numel(hyp.cov{jj})-1);
      pos = pos + numel(hyp.cov{jj});
    end
    val = negCrossEntropyFull(M,L,hyp,x);
  end
    
  function d = grad(theta, hyp, x)
    M = theta(1:Q*N)';
    % only the lower triangular are variables
    L = reshape(theta(Q*N+1:end),Q*N,N);
    trueL = L;
    for jj=1:Q
      start_row = (jj-1)*N+1;
      end_row = jj*N;
      Lj = tril(L(start_row:end_row,:));
      trueL(start_row:end_row,:) = Lj;
    end
    [~,dM,dL] = negCrossEntropyFull(M,trueL,hyp,x);
    d = [dM(:);dL(:)]';
  end

  function d = gradhyp(theta, M, L, hyp, x)
    pos = 1;
    for jj=1:Q
      hyp.cov{jj} = theta(pos:pos+numel(hyp.cov{jj})-1)';
      pos = pos + numel(hyp.cov{jj});
    end
    [~,~,~,dcell] = negCrossEntropyFull(M,L,hyp,x);
    d = [];
    for jj=1:Q
      d = [d, dcell{jj}'];
    end
  end

end


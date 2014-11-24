function chek_entropyFull()
%CHEK_ENTROPYFULL chek_entropyFull()
%   Gradient check for entropyFull().
clear functions
N = 3;
Q = 3;
m = rand(N,1);
A = rand(N,N);
Sigma = A*A';
Lj = jit_chol(Sigma)';
L = []; M = [];
for j=1:Q
  M = [M; m(:)];
  L = [L; Lj];
end
theta = [M(:); L(:)];

[mygrad, delta] = gradchek(theta', @f, @grad, Q, N);
disp('maxdiff = ')
disp(max(abs(delta)))

function val = f(theta, Q, N)
M = theta(1:Q*N)';
% only the lower triangular are variables
L = reshape(theta(Q*N+1:end),Q*N,N);
trueL = L;
for j=1:Q
  start_row = (j-1)*N+1;
  end_row = j*N;
  Lj = tril(L(start_row:end_row,:));
  trueL(start_row:end_row,:) = Lj;
end
val = entropyFull(M,trueL,Q);


function d = grad(theta, Q, N)
M = theta(1:Q*N)';
% only the lower triangular are variables
L = reshape(theta(Q*N+1:end),Q*N,N);
trueL = L;
for j=1:Q
  start_row = (j-1)*N+1;
  end_row = j*N;
  Lj = tril(L(start_row:end_row,:));
  trueL(start_row:end_row,:) = Lj;
end
[~,dM,dL] = entropyFull(M,trueL,Q);
d = [dM(:);dL(:)]';

function test_AdiagB(N)
%TEST_AdiagB test_AdiagB(N)
%   Test for AdiagB(). See also AdiagB.m.
if nargin == 0
  N = 1000;
end
A = rand(N,N);
b = rand(N,1);
B = diag(b);
tic;
expected = A*B;
fprintf('naive computation takes %f seconds\n', toc);
tic;
result = AdiagB(A,B); % also AdiagB(a,B);
fprintf('efficient computation takes %f seconds\n', toc);
assert(norm(abs(expected-result)) < 1e-10, 'test_AdiagB() failed');
disp('test_AdiagB() passed');
end


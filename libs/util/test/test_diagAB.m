function test_diagAB(N)
%TEST_DIAGAB test_diagAB(N)
%   Test for diagAB(). See also DIAGAB.m.
if nargin == 0
  N = 1000;
end
a = rand(N,1);
B = rand(N,N);
A = diag(a);
tic;
expected = A*B;
fprintf('naive computation takes %f seconds\n', toc);
tic;
result = diagAB(A,B); % also diagAB(a,B);
fprintf('efficient computation takes %f seconds\n', toc);
assert(norm(abs(expected-result)) < 1e-10, 'test_diagAB() failed');
disp('test_diagAB() passed');
end


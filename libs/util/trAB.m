function x = trAB(A,B)
%TRACEAB x = trAB(A,B)
%   
% Efficient computation for the trace of product of two matrices.
%
% Trung Nguyen
x = sum(sum(A.*B',2));
end

function C = AdiagB(A,B)
%ADIAGB C = AdiagB(A,B)
%   C = A*B where B is diagonal.
N = size(B,1);
C = A.*repmat(diag(B)',N,1);
end


function Kinv = invChol(L)
%KINV = INVCHOL(Linv)
% Computes K^{-1} from L where L is the cholesky decompotion of K,
% L = chol(K) or K = L'L
Kinv = L\(L'\eye(size(L))); % can also use solve_chol(L,eye(size(L)))
end

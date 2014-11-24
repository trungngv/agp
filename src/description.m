function description()
%
% 1. LATENT FUNCTION VALUES
% --------------------------
% Assume Q latent function with N data points, the latent values are
% represented as 
% F = [f_11 ... f_Q1]
%     [f_12 ... f_Q2]
%     [ .    .     .]
%     [ .    .     .]
%     [f_1N  .  f_QN]
% i.e. F = [f_1 ... f_Q]
% 
% The vectorial representation of all latent values
%  f = F(:)
% The matrix representation each column is a latent function
%  F = reshape(f,N,Q)
% Values of the j-th function
%  F(:,j)
% Values of the n-th observations
%  F(n,:)
% It is up to the user to define the likelihood p(y_n | F(n,:))
%
% 2. VARIATIONAL PARAMETERS
% -------------------------
% The matrix M of size dxK
%         [m_1 ... m_K]
% where column k is the mean of the k-th mixture component (for ALL of f)
%
% Similarly S of size dxK 
%         [s_1  s_2 ... s_K]
% where column k is the covariance diagonal of the k-th mixture component (for ALL of f)         
% 
% W = [w_1 ... w_K]' contains the mixture weights
%
% The vector of all variational parameters:
%    u = [M(:); S(:); W(:)]
%
% To get back M, W, S from u:
%   M = reshape(u(1:K*d),d,K);
%   S = reshape(u(K*d+1:2*K*d),d,K);
%   W = u(2*K*d+1:end);
%
% Many times we may need only one component, but note that m_k, s_k, f
% always have same order of elements.
end


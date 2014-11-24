function [fval,dM,dL,dw] = lowerboundEntropy(M,L,w)
%LOWERBOUNDENTROPY [fval,dM,dL,dw] = lowerboundEntropy(M,L,w)
%   The lower bound of entropy (Lent) and its derivatives wrt variational
%   parameters.
%
%   M(:,k) : the means of the k-th component
%   L(:,k) : the log square root of the covaraince diagonal of the k-th component
%   w(k) : the log weight of the k-th component
Km = size(M,2);
w = exp(w);
% L = log(diag(Sigma)) => Sigma2 = exp(2*L)
S = exp(2*L);

% naive computation
fval = 0;
logNkl = zeros(Km,Km);
logq = zeros(Km,1);
for k=1:Km
  for l=1:Km
    logNkl(k,l) = varDiagGaussian(M(:,k),M(:,l),log(sqrt(S(:,k)+S(:,l))));
  end
  logq(k) = logsum(log(w) + logNkl(k,:)');
  fval = fval - w(k)*logq(k);
end
if nargout >= 2
  dM = zeros(size(M));
  dL = zeros(size(L));
  dw = zeros(size(w));
  for k=1:Km
    for l=1:Km
      common = w(l)*(exp(logNkl(k,l)-logq(k))+exp(logNkl(k,l)-logq(l)));
      Skl = S(:,k)+S(:,l);
      Mkl = M(:,k)-M(:,l);
      dM(:,k) = dM(:,k) + common*Mkl./Skl;
      dL(:,k) = dL(:,k) + common*(1./Skl - (Mkl./Skl).^2).*S(:,k);
    end
    dM(:,k) = w(k)*dM(:,k);
    dL(:,k) = w(k)*dL(:,k);
    dw(k) = -w(k)*(logq(k) + sum(w.*exp(logNkl(k,:)'-logq)));
  end
end
end


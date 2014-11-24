function plotIntensity(xyear,y,fmu,fvar,q5,q95)
%PLOTINTENSITY plotIntensity(xyear,y,fmu,fvar,q5,q95)
%   
figure; plotMeanAndStd(xyear,fmu,2*sqrt(fvar),[7 7 7]/8);
%maxf = max(fmu + 2*sqrt(fvar)) + 0.1;
maxf = 0.6;
width = 0.05; % width corresponding to max intensity
ywidth = y/max(y)*width;
line([xyear,xyear],[maxf*ones(size(fmu)),maxf+ywidth],'color','k')
xlim([min(xyear),max(xyear)]);
if nargin > 4
  plot(xyear,q5,'m.','MarkerSize',0.1);
  plot(xyear,q95,'m.','MarkerSize',0.1);
end
ylim([0 0.7]);
title('Intensity');



function plotBox(points,methods,ftitle,ytitle,flegend,saveto,datalim)
%PLOTBOX plotBox(points,methods,ftitle,ytitle,flegend,saveto,datalim)
%
figure; hold off;
if isempty(datalim)
  %datalim = [quantile(points(:),0.025),quantile(points(:),0.975)];
  datalim = [min(min(points)),quantile(points(:),0.975)];
end

%datalim = [-10,10];
boxplot(points,'whisker',1.5,'colors','k','extrememode','clip',...
  'datalim',datalim,'symbol','.k','outliersize',2);
title(ftitle,'FontSize',24);
set(gca, 'XTick', 1:numel(methods), 'XTickLabel', methods);
set(gca, 'FontSize', 24);
ylabel(ytitle);
if ~isempty(flegend)
  legend(flegend,'Location','Best');
end
if ~isempty(saveto)
  saveas(gcf, saveto,'epsc');
end
end


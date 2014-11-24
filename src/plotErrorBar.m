function plotErrorBar(means,stds,methods,ftitle,ytitle,flegend,saveto)
%PLOTERRORBAR plotErrorBar(means,stds,methods,ftitle,ytitle,flegend,saveto)
%
figure; hold off;
errorbar(means, stds, 'sk','MarkerFaceColor','k');
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


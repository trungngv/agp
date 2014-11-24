W = what('src/test');
numfiles = numel(W.m);
for i=1:numfiles
  if strfind(W.m{i},'chek') == 1
    dotpos = strfind(W.m{i},'.m');
    fname = W.m{i}(1:dotpos-1);
    disp(['press any key to check ::: ' fname])
    pause
    feval(fname);
    fprintf('done checking... press any key \n')
    pause
  end
end
disp('well done')


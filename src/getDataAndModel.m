function [x,y,xt,yt,model] = getDataAndModel(data)
%GETDATAANDMODEL [x,y,xt,yt,model] = getDataAndModel(data)
%   Returns one of the datasets for demo
xt = []; yt = [];
if data <= 2
  model = 'classification';
elseif data <= 6
  model = 'regression';
elseif data <= 10
  model = 'warp';
else
  model = 'multiclass';
end

switch data
  case 1
    load toy
  case 2
    load iris
    indice = (Y == 1 | Y == 2);
    y = Y(indice);
    y(y == 2) = -1;
    x = X(indice,:);
    x = standardize(x,[],[]);
    indice = randperm(size(x,1));
    ntrain = 80;
    xt = x(indice(ntrain+1:end),:);
    yt = y(indice(ntrain+1:end));
    x = x(indice(1:ntrain),:);
    y = y(indice(1:ntrain));
  case 3
    x = linspace(-5,5,100)';
    %x = linspace(-1,1,10)';
    y = sin(x) + 0.01*randn(size(x));
  case 4
    x = linspace(-5,5,100)';
    y = sin(x) + cos(1.5*x) + 0.1*randn(size(x));
  case 5
    x = linspace(-5,5,100)';
    y = sample_gp(x,'covSEard',[0.01,1],1);
  case 6
    load motorcycle
    yt = y; xt = x;
    %[y,ymean,ystd] = standardize(y,[],[]);
  case 7
    % warp likelihood
    rng(1111,'twister');
    N = 101;
    x = linspace(-pi,pi,N)';
    y = (sin(x) + 0.1*randn(N,1)).^3;
    xt = linspace(-pi,pi,N-1)';
    yt = (sin(xt) + 0.1*randn(N-1,1)).^3;
    save('data/warp');
  case 8
    % creep
    rng(111000,'twister');
    data = load('creep');
    ally = data(:,2);
    allx = data(:,[1,3:end]);
    Ntrain = 800;
    randind = randperm(size(data,1));
    x = allx(randind(1:Ntrain),:);
    xt = allx(randind(Ntrain+1:end),:);
    y = ally(randind(1:Ntrain),:);
    yt = ally(randind(Ntrain+1:end),:);
  case 9
    % abalone
    rng(111000,'twister');
    data = load('abalone');
    ally = data(:,end);
    allx = data(:,1:end-1);
    Ntrain = 1000;
    randind = randperm(size(data,1));
    x = allx(randind(1:Ntrain),:);
    xt = allx(randind(Ntrain+1:end),:);
    y = ally(randind(1:Ntrain),:);
    yt = ally(randind(Ntrain+1:end),:);
  case 10
    % ailerons
    rng(111000,'twister');
    data = load('ailerons.data');
    ally = data(:,end);
    allx = data(:,1:end-1);
    Ntrain = 1000;
    randind = randperm(size(data,1));
    x = allx(randind(1:Ntrain),:);
    xt = allx(randind(Ntrain+1:end),:);
    y = ally(randind(1:Ntrain),:);
    yt = ally(randind(Ntrain+1:end),:);
  case 11
    % toy multi-class classification
    % linearly separable classes:
    % (0,1) : class 1, (2,3) : class 2, (4:5) : class 3
    rng(1234,'twister');
    n1 = 50; n2 = 50; n3 = 50;
    x1 = rand(n1,1);  y1 = repmat([1,0,0],size(x1,1),1);
    x2 = 1.1+rand(n2,1); y2 = repmat([0,1,0],size(x2,1),1);
    x3 = 2.1+rand(n3,1); y3 = repmat([0,0,1],size(x3,1),1);
    ntrain = 100;
    ntest = 50;
    allx = [x1; x2; x3];
    ally = [y1; y2; y3];
    % shuffle
    ind = randperm(size(allx,1));
    allx = allx(ind,:);
    ally = ally(ind,:);
    % split into training / testing
    x = allx(1:ntrain,:);
    y = ally(1:ntrain,:);
    xt = allx(ntrain+1:end,:);
    yt = ally(ntrain+1:end,:);
  case 12
    % glass data for multi-class classification
    rng(1234,'twister');
    glass = load('uci.glass.data');
    allx = glass(:,2:end-1);
    allx = standardize(allx,[],[]);
    ally = glass(:,end);
    idx = find(ally>4); % because none in the data is in class 4
    ally(idx) = ally(idx)-1;
    % convert labels into one-of-K
    ally = oneOfK(ally,max(ally));
    N = size(allx,1);
    ntrain = round(0.8*N);
    % shuffle
    ind = randperm(N);
    allx = allx(ind,:);
    ally = ally(ind,:);
    % split into training / testing
    x = allx(1:ntrain,:);
    y = ally(1:ntrain,:);
    xt = allx(ntrain+1:end,:);
    yt = ally(ntrain+1:end,:);
end
end


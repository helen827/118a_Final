clc;
clear all;


X = covTrainData(:,1:54);
Y = covTrainData(:,55);

%% 2 decision tree
Md1 = fitctree (X, Y);

%% cross validation
[~,~,~,bestlevel] = cvLoss(Md1,'SubTrees','All');

%% minleafsize
leafs = logspace(1,2,10);
rng('default')
N = numel(leafs);
err = zeros(N,1);
for n=1:N
    t = fitctree(X,Y,'CrossVal','On',...
        'MinLeafSize',leafs(n));
    err(n) = kfoldLoss(t);
end
[minVal minInd]=min(err);
bestLeafSize = leafs(minInd);
plot(leafs,err);
xlabel('Min Leaf Size');
ylabel('cross-validated error');

%% plug in bestLeafSize for an optimal tree
OptimalTree = fitctree(X,Y,'MinLeafSize',bestLeafSize);
view(OptimalTree,'mode','graph')
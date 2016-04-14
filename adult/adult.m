%% train data

[Numbers, Strings, Data] = xlsread('adult.csv');
cleanTrainData = [];
maxNum = 0;

for featureOrder = [2 4 6 7 8 9 10 14 15]
    adultData = double(nominal(Data(:,featureOrder)));
    for i = 1 : max(adultData)
        cleanTrainData(:, i + maxNum) = adultData==i;
    end
    maxNum = maxNum + max(adultData);
end

AdultTrainData = datasample(cleanTrainData,5000,'Replace',false);

data = [Numbers(:,1) Numbers(:,3) Numbers(:,5) Numbers(:,11) Numbers(:,12) Numbers(:,13) cleanTrainData(:,1:(size(cleanTrainData,2)-2))];
target = cleanTrainData(:,(size(cleanTrainData,2)-1):(size(cleanTrainData,2)));

%%
% test data
[Numbers, Strings, Data] = xlsread('adultTest.csv');
cleanTestData = [];
maxNum = 0;

for featureOrder = [2 4 6 7 8 9 10 14 15]
    adultTestData = double(nominal(Data(:,featureOrder)));
    for i = 1 : max(adultTestData)
        cleanTestData(:, i + maxNum) = adultTestData==i;
    end
    maxNum = maxNum + max(adultTestData);
end
AdultTestData = datasample(cleanTestData,5000,'Replace',false);

%% 
   trainLabel = AdultTrainData(:,103); %last two column contains labels
   trainData = AdultTrainData(:,1:101); %next 13 columns are data
   
   testLabel = AdultTestData(:,103);
   testData = AdultTestData(:,1:101);
   
   trainData = trainData - repmat(mean(trainData,1),size(trainData,1),1); %center
   trainData = trainData./repmat(sqrt(var(trainData)),size(trainData,1),1); %normalize
   
   testData = testData - repmat(mean(testData,1),size(testData,1),1); %center
   testData = testData./repmat(sqrt(var(testData)),size(testData,1),1); %normalize
   testSize =  size(cleanTestData,1);
   TrainMean = mean(trainData,1);
   TrainStd = std(trainData,1,1);
   TestMean = mean(testData,1);
   TestStd = std(testData,1,1);

%% normalize
  mu = mean(data);
  sigma = std(data);
  muMat = repmat(mu,size(data,1),1); 
  sigMat = repmat(sigma,size(data,1),1); 
  data_norm = (data - muMat )./sigMat ; 
  data = [data_norm target];
  data = [data_norm target];

trainsample = datasample(data,5000,1);

testset = data(~ismember(data,trainsample,'rows'),:);

   
%% knn
knn(3)
  

%% Decision Tree
X = AdultTrainData(:,1:101);
Y = AdultTrainData(:,103); 

Md1 = fitctree (X, Y);

% cross validation
[~,~,~,bestlevel] = cvLoss(Md1,'SubTrees','All');

% minleafsize
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

% plug in bestLeafSize for an optimal tree
OptimalTree = fitctree(X,Y,'MinLeafSize',bestLeafSize);
view(OptimalTree,'mode','graph')


%% neural network

t = target';

% normalize data
mu = mean(data);
sigma = std(data);
muMat = repmat(mu,size(data,1),1); 
sigMat = repmat(sigma,size(data,1),1); 
data_norm = (data - muMat )./sigMat ; 
x = data_norm';

% Pattern Recognition with a Neural Network

setdemorandstream(391418381)

net = patternnet(100);

[net,tr] = train(net,x,t);

testX = x(:,tr.testInd);

testT = t(:,tr.testInd);

testY = net(testX);

testIndices = vec2ind(testY);

[c,cm] = confusion(testT,testY)



fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));

fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

plotroc(testT,testY)

%Percentage Correct Classification   : 83.619984%

% Percentage Incorrect Classification : 16.380016%


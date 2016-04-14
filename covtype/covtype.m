data = importdata('covtype.data');

%% train data
% without replacement 
covTrainData = datasample(data,5000,'Replace',false);
covTestData = data(~ismember(data,covTrainData,'rows'),:);



%% knn
knn(3);
  

%% Decision Tree

%% neural network

dataTrain = data(1:(size(data,1)-1),:);
instance = dataTrain(:,1:(size(dataTrain,2)-1));
target = dataTrain(:,(size(dataTrain,2)));
target = target==mode(target);
nonzero = find(mean(instance));
nonzeroins = instance(:,nonzero);
 % normalize data
    mu = mean(nonzeroins);
    sigma = std(nonzeroins);
    muMat = repmat(mu,size(nonzeroins,1),1); 
    sigMat = repmat(sigma,size(nonzeroins,1),1); 
    data_norm = (nonzeroins - muMat )./sigMat ; 
    %% training sample 
data = [data_norm target];
trainsample = datasample(data,5000,1);
testset = data(~ismember(data,trainsample,'rows'),:);
%% neural network 
t = target';
x = data_norm';
setdemorandstream(391418381)
net = patternnet(10);
[net,tr] = train(net,x,t);
testX = x(:,tr.testInd);
testT = t(:,tr.testInd);
testY = net(testX);
testIndices = vec2ind(testY);
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);
plotroc(testT,testY)



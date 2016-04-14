%function [linear, RBF ] = svm(k, AdultTrainData,AdultTestData)
load 'adultTrainData.mat';
load 'adultTestData.mat';


k = 1
%% a
[trainIndex, testIndex] = dividerand(length(AdultTrainData), 0.8, 0.2);

trainX = AdultTrainData(:,1:101);
trainY = AdultTrainData(:,103)
testX = AdultTestData(:,1:101);
testY = AdultTestData(:,103);    


%% b. train a SVMs classifer using a linear kernel

C = 10.^[-7:5];
gamma =  [0.001 0.005 0.01 0.05 0.1 0.5 1.2];


linear_accuracy = zeros(length(C),1);
%%
for i = 1:length(C)
    linear_accuracy(i) = svmtrain(trainY, trainX, sprintf('-t 0 -v 5 -c %f -q',C(i));
end

%%
[C_value, C_index] = max(linear_accuracy);

C_value_bestfit(k) = C(C_index); %find the largest accuracy by find the loss

best_linear_model = svmtrain(trainY, trainX,sprintf('-c %d -t 0 -v 5', C_value_bestfit(k)) );
[~,accuracy_linear,prob_linear] = svmpredict(testY, testX, best_linear_model);

linear(k) = accuracy_linear(1); % accuracy matrix

%% c. train a SVM classifer using the radial basis function kernel


RBF_accuracy = zeros(length(C),length(gamma));

for i = 1:length(C)
    for j = 1:length(gamma)
        RBF_accuracy(i,j) = svmtrain(trainY, trainX,sprinf('-t 2, -v 5 -c %f -g %f -q', C(i), gamma(j)));
    end
end

[C_value, C_index] = max(RBF_accuracy);
[gamma_value, gamma_index] = max(RBF_accuracy);

gamma_value_bestfit(k) = gamma(gamma_index);

best_RBF_model = svmtrain(trainY, trainX,sprinf('-t 2, -v 5 -c %f -g %f -q', C(i), gamma_value));
[~, accuracy_RBF, prob_RBF] = svmpredit(testY,testX,best_RBF_model);
RBF(k) = accuracy_RBF(1);

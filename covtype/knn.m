function [accuracy] = knn(k)
  
load 'covTestData.mat'
load 'covTrainData.mat'
   accuracy = 0;
   trainLabel = covTrainData(:,55); 
   trainData = covTrainData(:,1:54); 
   
   testLabel = covTestData(:,55);
   testData = covTestData(:,1:54);
   
   trainData = trainData - repmat(mean(trainData,1),size(trainData,1),1); %center
   trainData = trainData./repmat(sqrt(var(trainData)),size(trainData,1),1); %normalize
   
   testData = testData - repmat(mean(testData,1),size(testData,1),1); %center
   testData = testData./repmat(sqrt(var(testData)),size(testData,1),1); %normalize
   testSize =  size(covTestData,1);
   TrainMean = mean(trainData,1);
   TrainStd = std(trainData,1,1);
   TestMean = mean(testData,1);
   TestStd = std(testData,1,1);
   
   
   %For each point in Test Set.
   for row = 1:size(testData,1)
      %Compute its euclidean distance to each point in Train Set.
      %(Hint use repmat to do this step in one command, without for loop).
      distance = sqrt(sum(bsxfun(@minus, testData(row,:), trainData).^2, 2));

      [distance position] = sort(distance);
      for i = k
      distance = distance(1:i);
      position = position(1:i);
      
      %Take the mode of the k-nearest neighbors     
      labels = trainLabel(position);     
      prediction = mode(labels);
     
      %Compare it compare the predicted label with the true label
      %if they are equal
         % add 1 to the accuracy
      end
  
         if prediction == testLabel(row)
          accuracy = accuracy + 1;
      end
      
   end
   
   %return accuracy divided by TestSize
   accuracy = accuracy / testSize;
end
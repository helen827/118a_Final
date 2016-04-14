function [accuracy] = KNearestN(k, Filename)

   RawData = dlmread(Filename,',');
   
   k = 1;
   numpoints = size (RawData,1);
   m = RawData(:,1);

   accuracy = 0;
  
   AllLabels = RawData(:,1); %first column contains labels
   AllData = RawData(:,2:end); %next 13 columns are data
   
   AllData = AllData - repmat(mean(AllData,1),size(AllData,1),1); %center
   AllData = AllData./repmat(sqrt(var(AllData)),size(AllData,1),1); %normalize
   
   randNums = randperm(size(AllLabels,1));
   
   TestSize = ceil (numpoints * 0.3);
   TrainSize = floor (numpoints * 0.7);

   TestData = AllData(randNums(1:TestSize),:);     %this contains Xi
   TestLabels = AllLabels(randNums(1:TestSize)); %this contains Yi
   
   
   TrainData = AllData(randNums((TestSize+1):end),:);
   TrainLabels = AllLabels(randNums((TestSize+1):end));
   
   TestMean = mean(TestData,1);
   TestStd = std(TestData,1,1);
   
   
   %For each point in Test Set.
   for row = 1:size(TestData,1)
      %Compute its euclidean distance to each point in Train Set.
      %(Hint use repmat to do this step in one command, without for loop).
      distance = sqrt(sum(bsxfun(@minus, TestData(row,:), TrainData).^2, 2));

      [distance position] = sort(distance);
      distance = distance(1:k);
      position = position(1:k);
      
      %Take the mode of the k-nearest neighbors     
      labels = TrainLabels(position);     
      prediction = mode(labels);
     
      %Compare it compare the predicted label with the true label
      %if they are equal
         % add 1 to the accuracy
      if prediction == TestLabels(row)
          accuracy = accuracy + 1;
      end
      
   end
   
   %return accuracy divided by TestSize
   accuracy = accuracy / TestSize;
   
   
   end
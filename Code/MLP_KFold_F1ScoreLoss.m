%This function performs KFold Cross validation ( with minority class upsampling
%for training data) for MLP and returns the mean F1 Score Error

function meanF1ScoreError = MLP_KFold_F1ScoreLoss(x, t, k,networkDepth, numHiddenNeurons, lr, momentum, trainFcn, transferFcn)
    % creating the network based on input parameters    
    hiddenLayerSize = numHiddenNeurons * ones(1, networkDepth);
    net = patternnet(hiddenLayerSize, char(trainFcn));
    net.trainParam.epochs = 100; 
    net.trainParam.max_fail = 6; 
    net.trainParam.lr = lr; %Learning rate
    net.trainParam.mc = momentum;%Maximum validation failures

    for i = 1:networkDepth 
        net.layers{i}.transferFcn = char(transferFcn); 
    end
    net.performFcn = 'crossentropy';

    kfold=k;
    rng(110);
    
    % Divide data into k-folds
    cv=cvpartition(t,'kfold',kfold,'Stratify', true); 

    % Save the performance in arrays
    accuracyFold=zeros(kfold,1); f1ScoreErrorFold=zeros(kfold,1); 
    
    %K Fold cross validation with minority class upsampling for each
    %training dataset
    for k=1:kfold
        xTrain=x(cv.training(k),:); yTrain=t(cv.training(k),:);
        xTest=x(cv.test(k),:); yTest=t(cv.test(k),:);
        
        %Oversampling of minority class in training data using Adaptive Synthetic Sampling Method for Imbalanced Data
        [out_featuresSyn, out_labelsSyn] = ADASYN(xTrain, yTrain, [], [], [], false);
        
        %Stitching the synthesized data with original dataset from
        %training.
        xTrain=[xTrain;out_featuresSyn];
        yTrain=[yTrain;out_labelsSyn];
        yTrain=dummyvar(categorical(yTrain));
        yTest=dummyvar(categorical(yTest));
              
        % Train Network
        net = train(net, xTrain', yTrain');

        % Evaluate on validation set and compute F1 Score Error
        y = net(xTest');
        
       [~,con]=confusion(yTest',y); 
       
       %Calculating the performance metrics based on confusion matrix
       [accuracyMLP,precisionMLP, recallMLP, specificityMLP,fscoreMLP] = PerformanceMetrics(con);

       accuracyFold(k)=accuracyMLP;
       %We are using F1 score error as error
       f1ScoreErrorFold(k)=1-fscoreMLP;
    end;
    
    %Calculating Mean score across folds
    meanAccuracy=mean(accuracyFold);
    meanF1ScoreError=mean(f1ScoreErrorFold);

end
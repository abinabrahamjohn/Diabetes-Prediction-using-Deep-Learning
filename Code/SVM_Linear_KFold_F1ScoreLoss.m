%This function performs KFold Cross validation ( with minority class upsampling
%for training data) for SVM Linear Kernel and returns the mean F1 Score Error
function meanF1ScoreError = SVM_Linear_KFold_F1ScoreLoss(x, t,k, boxConstraint)
    kfold=k;
    rng(110);
    % Divide data into k-folds
    cv=cvpartition(t,'kfold',kfold,'Stratify', true); 

    % Prediction arrays
    accuracyFold=zeros(kfold,1); f1ScoreErrorFold=zeros(kfold,1); 
    
    %K Fold cross validation with minority class upsampling for each
    %training dataset
    for k=1:kfold
        xTrain=x(cv.training(k),:); yTrain=t(cv.training(k),:);
        xTest=x(cv.test(k),:); yTest=t(cv.test(k),:);
        
        [out_featuresSyn, out_labelsSyn] = ADASYN(xTrain, yTrain, [], [], [], false);
        %Stitching the synthesized data with original dataset from training.
        xTrain=[xTrain;out_featuresSyn];
        yTrain=[yTrain;out_labelsSyn];
        
        %create and train SVM with linear kernel model
        svm_model=fitcsvm(...
        xTrain, ...
        yTrain, ...
        'KernelFunction', 'linear', ...
        'BoxConstraint', boxConstraint, ...
        'Standardize', true, ...
        'ClassNames', [0; 1]);
    
        % Evaluate on validation set and compute F1 Score Error
        svm_prediction = predict(svm_model, xTest);

        con=confusionmat(yTest',svm_prediction'); 

        [accuracySVM,precisionSVM, recallSVM, specificitySVM,fscoreSVM] = PerformanceMetrics(con);
        accuracyFold(k)=accuracySVM;
        f1ScoreErrorFold(k)=1-fscoreSVM;
    end;
    %Calculating Mean accuracy and F1score across folds
    meanAccuracy=mean(accuracyFold);
    meanF1ScoreError=mean(f1ScoreErrorFold);

end
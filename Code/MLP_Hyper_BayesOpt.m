%Performs hyperparameter tuning using bayesian optimization with input
%range for parameters based on results from grid search

clc; clear; close all;

%Loading training dataset
load('..\Data\dataTrain');

features = dataTrain(:, 1:end-1);
labels = dataTrain(:, end);
kfold=10;
rng(110);

%Here the range was determined based on results we got from grid search
vars = [optimizableVariable('networkDepth', [1 2], 'Type', 'integer');
        optimizableVariable('numHiddenNeurons', [20, 35], 'Type', 'integer');
	    optimizableVariable('lr', [1e-2 1], 'Transform', 'log');
        optimizableVariable('momentum', [1e-2 1]);
        optimizableVariable('trainFcn', {'traingd', 'traingdm','trainscg','traingdx'}, 'Type', 'categorical');
        optimizableVariable('transferFcn', {'logsig','tansig','softmax'}, 'Type', 'categorical')];
        %optimizableVariable('transferFcn', {'logsig', 'poslin', 'tansig', 'purelin','softmax'}, 'Type', 'categorical')];

%The objective function does a KFold Cross valdiation (with minority class
%upsampling on training data)and returns the average F1score.
minfn = @(T)MLP_KFold_F1ScoreLoss(features, labels,kfold,T.networkDepth, T.numHiddenNeurons,...
    T.lr, T.momentum, T.trainFcn, T.transferFcn);
results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 20);
T = bestPoint(results);

%Storing thetop 5 results and hyperparameters obtained
MLP_Tuning_Results=[results.XTrace array2table(results.ObjectiveTrace,'VariableNames',{'ErrorObjective'})];
MLP_Tuning_Results_Sorted=sortrows(MLP_Tuning_Results,'ErrorObjective');
MLP_Tuning_Results_Top5=MLP_Tuning_Results_Sorted(1:5,:);
save('..\Results\MLP_Tuning_Results_Top5','MLP_Tuning_Results_Top5');
%%
%Here we determine the best 2 values of the Bayesian Optimization objective function(Lowest F1 Score Error) 
%and their respective combinations of tuning parameters to build models and
%train them on the full training dataset

%MLP Tuned Model 1 - Model with best results in bayesian optimization hyper
%parameter tuning

hiddenLayerSize = ones(1, T.networkDepth) * T.numHiddenNeurons;
mlpTunedModel1 = patternnet(hiddenLayerSize, char(T.trainFcn));
mlpTunedModel1.trainParam.lr = T.lr; 
mlpTunedModel1.trainParam.mc = T.momentum; 
mlpTunedModel1.performFcn = 'crossentropy';
mlpTunedModel1.divideParam.trainRatio = 0.8;
mlpTunedModel1.divideParam.valRatio   = 0.2;
mlpTunedModel1.divideParam.testRatio  = 0;

for i = 1:T.networkDepth
    mlpTunedModel1.layers{i}.transferFcn = char(T.transferFcn);
end
  labels = dummyvar(categorical(labels));
 %Training model on full training set
[mlpTunedModel1,tr1] = train(mlpTunedModel1, features', labels');
%Error curve (Cross entropy) for the best model
figure;
plotperform(tr1)
save('..\Models\mlpTunedModel1','mlpTunedModel1');

%%
%MLP Tuned Model 2- Model with 2nd best results in bayesian optimization hyper
%parameter tuning
hiddenLayerSize = ones(1, MLP_Tuning_Results_Top5.networkDepth(2)) * MLP_Tuning_Results_Top5.numHiddenNeurons(2);
mlpTunedModel2 = patternnet(hiddenLayerSize, char(MLP_Tuning_Results_Top5.trainFcn(2)));
mlpTunedModel2.trainParam.lr = MLP_Tuning_Results_Top5.lr(2); 
mlpTunedModel2.trainParam.mc = MLP_Tuning_Results_Top5.momentum(2); 
mlpTunedModel2.performFcn = 'crossentropy';

for i = 1:T.networkDepth
    mlpTunedModel2.layers{i}.transferFcn = char(MLP_Tuning_Results_Top5.transferFcn(2));
end
[mlpTunedModel2,tr2] = train(mlpTunedModel2, features', labels');
save('..\Models\mlpTunedModel2','mlpTunedModel2');

%%
%MLP Tuned Model 3 - Model 1 trained with noise
%Adding noise
features=0.025*randn(size(features)) + features;

hiddenLayerSize = ones(1, T.networkDepth) * T.numHiddenNeurons;
mlpTunedModel1_Noise = patternnet(hiddenLayerSize, char(T.trainFcn));
mlpTunedModel1_Noise.trainParam.lr = T.lr; 
mlpTunedModel1_Noise.trainParam.mc = T.momentum; 
mlpTunedModel1_Noise.performFcn = 'crossentropy';
mlpTunedModel1_Noise.divideParam.trainRatio = 0.8;
mlpTunedModel1_Noise.divideParam.valRatio   = 0.2;
mlpTunedModel1_Noise.divideParam.testRatio  = 0;

for i = 1:T.networkDepth
    mlpTunedModel1_Noise.layers{i}.transferFcn = char(T.transferFcn);
end
 %labels = dummyvar(categorical(labels));
 %Training model on full training set
[mlpTunedModel1_Noise,tr1] = train(mlpTunedModel1_Noise, features', labels');
%Error curve (Cross entropy) for the best model
figure;
plotperform(tr1)
save('..\Models\mlpTunedModel1_Noise','mlpTunedModel1_Noise');

%%
%MLP Tuned Model 4 - Model 1 trained with noise and minority class oversampled using
%ADASYN
%Oversampling of minority class in training data using Adaptive Synthetic Sampling Method for Imbalanced Data
hiddenLayerSize = ones(1, T.networkDepth) * T.numHiddenNeurons;
mlpTunedModel1_Noise_ADASYN = patternnet(hiddenLayerSize, char(T.trainFcn));
mlpTunedModel1_Noise_ADASYN.trainParam.lr = T.lr; 
mlpTunedModel1_Noise_ADASYN.trainParam.mc = T.momentum; 
mlpTunedModel1_Noise_ADASYN.performFcn = 'crossentropy';
mlpTunedModel1_Noise_ADASYN.divideParam.trainRatio = 0.8;
mlpTunedModel1_Noise_ADASYN.divideParam.valRatio   = 0.2;
mlpTunedModel1_Noise_ADASYN.divideParam.testRatio  = 0;

for i = 1:T.networkDepth
    mlpTunedModel1_Noise_ADASYN.layers{i}.transferFcn = char(T.transferFcn);
end
labels = dataTrain(:, end);
[out_featuresSyn, out_labelsSyn] = ADASYN(features, labels, [], [], [], false);

%Stitching the synthesized data with original dataset from training.
features=[features;out_featuresSyn];
labels=[labels;out_labelsSyn];
labels=dummyvar(categorical(labels));
[mlpTunedModel1_Noise_ADASYN,tr1] = train(mlpTunedModel1_Noise_ADASYN, features', labels');
%Error curve (Cross entropy) for the best model
figure;
plotperform(tr1)
save('..\Models\mlpTunedModel1_Noise_ADASYN','mlpTunedModel1_Noise_ADASYN');

%%
%MLP Tuned Model 5 - Model 1 trained with minority class oversampled using
%ADASYN
%Oversampling of minority class in training data using Adaptive Synthetic Sampling Method for Imbalanced Data
hiddenLayerSize = ones(1, T.networkDepth) * T.numHiddenNeurons;
mlpTunedModel1_ADASYN = patternnet(hiddenLayerSize, char(T.trainFcn));
mlpTunedModel1_ADASYN.trainParam.lr = T.lr; 
mlpTunedModel1_ADASYN.trainParam.mc = T.momentum; 
mlpTunedModel1_ADASYN.performFcn = 'crossentropy';
mlpTunedModel1_ADASYN.divideParam.trainRatio = 0.8;
mlpTunedModel1_ADASYN.divideParam.valRatio   = 0.2;
mlpTunedModel1_ADASYN.divideParam.testRatio  = 0;

for i = 1:T.networkDepth
    mlpTunedModel1_ADASYN.layers{i}.transferFcn = char(T.transferFcn);
end
features = dataTrain(:, 1:end-1);
labels = dataTrain(:, end);
[out_featuresSyn, out_labelsSyn] = ADASYN(features, labels, [], [], [], false);

%Stitching the synthesized data with original dataset from training.
features=[features;out_featuresSyn];
labels=[labels;out_labelsSyn];
labels=dummyvar(categorical(labels));
[mlpTunedModel1_ADASYN,tr1] = train(mlpTunedModel1_ADASYN, features', labels');
%Error curve (Cross entropy) for the best model
figure;
plotperform(tr1)
save('..\Models\mlpTunedModel1_ADASYN','mlpTunedModel1_ADASYN');
%%
labels = dataTrain(:, end);
meanF1ScoreError = MLP_KFold_F1ScoreLoss(features, labels, kfold,T.networkDepth, T.numHiddenNeurons, T.lr, T.momentum, T.trainFcn, T.transferFcn)

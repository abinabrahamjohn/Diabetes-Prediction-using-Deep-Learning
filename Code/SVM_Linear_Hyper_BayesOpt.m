%Performs hyperparameter tuning using bayesian optimization with input
%range for parameters based on results from grid search
clear;  close all;clc;

%Loading training dataset
load('..\Data\dataTrain');

input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

kfold=10;

%Here the range was determined based on results we got from grid search
vars = optimizableVariable('boxConstraint',[0.05,0.2],'Transform','log');
	
%The objective function does a KFold Cross valdiation (with minority class
%upsampling on training data)and returns the average F1score.
minfn = @(T)SVM_Linear_KFold_F1ScoreLoss(input, target, kfold,T.boxConstraint);
results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 50);
T = bestPoint(results);

%Storing thetop 5 results and hyperparameters obtained
SVM_Linear_Tuning_Results=[results.XTrace array2table(results.ObjectiveTrace,'VariableNames',{'ErrorObjective'})];
SVM_Linear_Tuning_Results_Sorted=sortrows(SVM_Linear_Tuning_Results,'ErrorObjective');
SVM_Linear_Tuning_Results_Top5=SVM_Linear_Tuning_Results_Sorted(1:5,:);
save('..\Results\SVM_Linear_Tuning_Results_Top5','SVM_Linear_Tuning_Results_Top5');
svm_linear_bestmodel_kfoldF1ScoreError=results.MinObjective
save('..\Results\KFoldCrossValidation','svm_linear_bestmodel_kfoldF1ScoreError');

%Here we determine the best values of the Bayesian Optimization objective function(Lowest F1 Score Error) 
%and their respective combinations of tuning parameters to build models and
%train them on the full training dataset

%Tuned Model trained on whole training data with oversamplign of minority
%class
[out_featuresSyn, out_labelsSyn] = ADASYN(input, target, [], [], [], false);
%Stitching the synthesized data with original dataset from training.
input=[input;out_featuresSyn];
target=[target;out_labelsSyn];
svm_linear=fitcsvm(...
            input, ...
            target, ...
            'KernelFunction', 'linear', ...
            'BoxConstraint', T.boxConstraint, ...
            'Standardize', true, ...
            'ClassNames', [0; 1]);
svm_linear=fitPosterior(svm_linear,input,target)
save('..\Models\svm_linear','svm_linear');
%Performs hyperparameter tuning using bayesian optimization with input
%range for parameters based on results from grid search
clear all; clc; close all;

%Loading training dataset


load('..\Data\dataTrain');

input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

kfold=3;

%Here the range was determined based on results we got from grid search
vars = [optimizableVariable('boxConstraint',[0.1,2],'Transform','log');
        optimizableVariable('polynomialOrder',[2 3],'Type','integer');
	   ];
%The objective function does a KFold Cross valdiation (with minority class
%upsampling on training data)and returns the average F1score.
minfn = @(T)SVM_Poly_KFold_F1ScoreLoss(input, target,kfold, T.boxConstraint,T.polynomialOrder);
results = bayesopt(minfn, vars,'IsObjectiveDeterministic', false,...
    'AcquisitionFunctionName', 'expected-improvement-plus', ...
    'MaxObjectiveEvaluations', 20);
T = bestPoint(results);

SVM_Poly_Tuning_Results=[results.XTrace array2table(results.ObjectiveTrace,'VariableNames',{'ErrorObjective'})];
SVM_Poly_Tuning_Results_Sorted=sortrows(SVM_Poly_Tuning_Results,'ErrorObjective');
SVM_Poly_Tuning_Results_Top5=SVM_Poly_Tuning_Results_Sorted(1:5,:);
save('..\Results\SVM_Poly_Tuning_Results_Top5','SVM_Poly_Tuning_Results_Top5');
svm_poly_bestmodel_kfoldF1ScoreError=results.MinObjective;
save('..\Results\KFoldCrossValidation','svm_poly_bestmodel_kfoldF1ScoreError');


%Here we determine the best values of the Bayesian Optimization objective function(Lowest F1 Score Error) 
%and their respective combinations of tuning parameters to build models and
%train them on the full training dataset

%Tuned Model trained on whole training data with oversamplign of minority
%class
[out_featuresSyn, out_labelsSyn] = ADASYN(input, target, [], [], [], false);
%Stitching the synthesized data with original dataset from training.
input=[input;out_featuresSyn];
target=[target;out_labelsSyn];
svm_poly=fitcsvm(...
            input, ...
            target, ...
            'KernelFunction', 'polynomial', ...
            'PolynomialOrder', T.polynomialOrder, ...
            'BoxConstraint', T.boxConstraint, ...
            'Standardize', true, ...
            'ClassNames', [0; 1]);
%SVMHyper_Poly_=[results.XTrace array2table(results.ObjectiveTrace)];
save('..\Models\svm_poly','svm_poly');
%Performs Grid search for hyper parameter tuning in SVM using Linear Kernel and 
%plots the time - hyperparameter relationship
clear all; clc; close all;

%Loading training dataset
load('..\Data\dataTrain');

input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

%Defining the number of folds for cross validation. Can be reduced for
%quicker implementation.
kfold=10;

%Hyperparameters with wide range for exploration in grid search
Box = [0.1 0.2 0.4 0.8 1.2 1.4 1.6 1.8 2 3];

%Defining array to store the grid search results.
Grid_Linear=[];

for i = 1:length(Box)
    %starting timer
    tic;
    %Determining k fold cross validation f1 score loss for the
    %selected hyper-parameter combination
    meanF1ScoreError = SVM_Linear_KFold_F1ScoreLoss(input, target, kfold,Box(i));
    %ending timer
    totalTime=toc;
    %Storing results
    Grid_Linear=[Grid_Linear;Box(i) meanF1ScoreError totalTime];
end
Grid_SVM_Linear=sortrows(Grid_Linear);
save('..\Results\Grid_SVM_Linear','Grid_SVM_Linear');

%Plotting Execution time v/s box constraint
figure;
plot(Grid_Linear(1:10,1),Grid_Linear(1:10,3),'Marker','*')
xlabel('Box Constraint');ylabel('Execution Time');
title('SVM Linear Time vs Box Constraint')

%Plotting Objective error v/s box constraint
figure;
plot(Grid_Linear(1:10,1),Grid_Linear(1:10,2),'Marker','*')
xlabel('Box Constraint');ylabel('Objective Error');
title('SVM Linear Objective Error vs Box Constraint')
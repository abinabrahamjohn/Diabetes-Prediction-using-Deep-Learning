%Performs Grid search for hyper parameter tuning in SVM using Polynomial Kernel and 
%plots the time - hyperparameter relationship
clear all; clc; close all;

%Loading training dataset
load('..\Data\dataTrain');

input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

%Defining the number of folds for cross validation. Reduced for
%quicker implementation.
kfold=3;

%Hyperparameters with wide range for exploration in grid search
Box = [ 0.1 0.2 0.4 0.8 1.2 1.4 1.6 1.8 2 3];
PolyOrder=[2 3 4];
Grid_Poly=[];

for i = 1:length(Box)
    for j = 1:length(PolyOrder)
        %starting timer
        tic;
        %Determining k fold cross validation f1 score loss for the
        %selected hyper-parameter combination
        meanF1ScoreError = SVM_Poly_KFold_F1ScoreLoss(input, target, kfold,Box(i),PolyOrder(j));
        %ending timer
        totalTime=toc;
        %Storing results
        Grid_Poly=[Grid_Poly;PolyOrder(j) Box(i) meanF1ScoreError totalTime];
    end
end
Grid_Poly=sortrows(Grid_Poly);
%Plotting Execution time v/s box constraint
figure;
plot(Grid_Poly(1:10,2),Grid_Poly(1:10,4),'-or',Grid_Poly(11:20,2),Grid_Poly(11:20,4),'-+g',Grid_Poly(21:30,2),Grid_Poly(21:30,4),'-xb')
xlabel('Box Constraint');ylabel('Execution Time');
title('SVM Polynomial Time Complexity-Box Constraint')
legend('Order=2','Order=3','Order=4')

%Plotting Objective error v/s box constraint
figure;
plot(Grid_Poly(1:10,2),Grid_Poly(1:10,3),'-or',Grid_Poly(11:20,2),Grid_Poly(11:20,3),'-+g',Grid_Poly(21:30,2),Grid_Poly(21:30,3),'-*b')
xlabel('Box Constraint');ylabel('ObjectiveError');
title('SVM Polynomial Error-Box Constraint')
legend('Order=2','Order=3','Order=4')
Grid_SVM_Poly=Grid_Poly;
save('..\Results\Grid_SVM_Poly','Grid_SVM_Poly');

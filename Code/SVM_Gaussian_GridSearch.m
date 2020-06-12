%Performs Grid search for hyper parameter tuning in SVM using Gaussian Kernel and 
%plots the time - hyperparameter relationship
clear all; clc; close all;

%Loading training dataset
load('..\Data\dataTrain');

input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

%Defining the number of folds for cross validation. Can be reduced for
%quicker implementation.
kfold=5;

%Hyperparameters with wide range for exploration in grid search
Box = [0.1 0.2 0.4 0.8 1.2 1.4 1.6 1.8 2 3];
Kernel = [1 2 3 4];
Grid_Gaussian=[];

for i = 1:length(Box)
    for j=1:length(Kernel)
        %starting timer
        tic;
        %Determining k fold cross validation f1 score loss for the
        %selected hyper-parameter combination
        meanF1ScoreError = SVM_Gaussian_KFold_F1ScoreLoss(input, target, kfold,Box(i),Kernel(j));
        totalTime=toc;
        Grid_Gaussian=[Grid_Gaussian;Box(i) Kernel(j) meanF1ScoreError totalTime];
    end
end
Grid_Gaussian=sortrows(Grid_Gaussian,2);
Grid_SVM_Gaussian=Grid_Gaussian;
save('..\Results\Grid_SVM_Gaussian','Grid_SVM_Gaussian');

%Plotting Box Constraint v/s execution time
figure;
plot(Grid_Gaussian(1:10,1),Grid_Gaussian(1:10,4),'-or',Grid_Gaussian(11:20,1),Grid_Gaussian(11:20,4),'-+g',Grid_Gaussian(21:30,1),Grid_Gaussian(21:30,4),'-*b',Grid_Gaussian(31:40,1),Grid_Gaussian(31:40,4),'-dk')
xlabel('Box Constraint');ylabel('Execution Time');
title('Gaussian Grid Time Complexity')
legend('KernelScale=1','KernelScale=2','KernelScale=3','KernelScale=4')

%Plotting Box Constraint v/s Objective error
figure;
plot(Grid_Gaussian(1:10,1),Grid_Gaussian(1:10,3),'-or',Grid_Gaussian(11:20,1),Grid_Gaussian(11:20,3),'-+g',Grid_Gaussian(21:30,1),Grid_Gaussian(21:30,3),'-*b',Grid_Gaussian(31:40,1),Grid_Gaussian(31:40,3),'-dk')
xlabel('Box Constraint');ylabel('Objective Error');
title('Gaussian Grid Error-Box Constraint')
legend('KernelScale=1','KernelScale=2','KernelScale=3','KernelScale=4')
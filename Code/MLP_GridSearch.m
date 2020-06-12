%Performs Grid search for hyper parameter tuning in MLP and 
%plots the time - hyperparameter relationship

clear all; clc; close all;

%Loading training dataset
load('..\Data\dataTrain');
input = dataTrain(:, 1:end-1);
target = dataTrain(:, end);

rng(110);

%Defining the number of folds for cross validation. Can be reduced for
%quicker implementation.
kfold=10;

%Hyperparameters with wide range for exploration in grid search
networkDepth = [1 2];
numHiddenNeurons = [10 20 30 40];
lr=[0.0001 0.001 0.01 0.1 0.9];
momentum=[0.01 0.1 0.5 1 2];

%Defining array to store the grid search results.
Grid_MLP=[];

for i = 1:length(networkDepth)
    for j=1:length(numHiddenNeurons)
        for k=1:length(lr)
            for l=1:length(momentum)
                %starting timer
                tic;
                %Determining k fold cross validation f1 score loss for the
                %selected hyper-parameter combination
                meanF1ScoreError = MLP_KFold_F1ScoreLoss(input, target,kfold,networkDepth(i), numHiddenNeurons(j),...
            lr(k), momentum(l), 'trainscg', 'tansig');
                %ending timer
                totalTime=toc;
                %Storing results
                Grid_MLP=[Grid_MLP;networkDepth(i) numHiddenNeurons(j)...
            lr(k) momentum(l) meanF1ScoreError totalTime];
            end
        end
    end
end
%Plotting 

%Sorting rows based on on number of hidden neurons
Grid_MLP=sortrows(Grid_MLP,2);

%Plotting Objective error v/s number of hidden neurons for constant network depth,
%learning rate and momentum. This is a simplfied version to understand the
%general trend a may have different trends for other learning rates,
%momentum
figure;
plot(Grid_MLP(1:50:151,2),Grid_MLP(1:50:151,5),'Marker','*')
xlabel('Hidden Neurons');ylabel('Objective Error');
title('MLP Grid Search :Objective Error v/s Hidden Neurons')

%Plotting Execution Time for KFold cross validation v/s number of hidden neurons for constant network depth,
%learning rate and momentum
figure;
plot(Grid_MLP(1:50:151,2),Grid_MLP(1:50:151,6),'Marker','*')
xlabel('Hidden Neurons');ylabel('Time');
title('MLP Grid Search :Time v/s Hidden Neurons')

%Plotting Execution Time for KFold cross validation v/s momentum for constant network depth,
%number of hidden neurons and learning rate
figure;
plot(Grid_MLP(1:5,4),Grid_MLP(1:5,6),'Marker','*')
xlabel('Momentum');ylabel('Time');
title('MLP Grid Search: Time v/s Momentum')

Grid_MLP=sortrows(Grid_MLP,4);

%Plotting Execution Time for KFold cross validation v/s learning rate for constant network depth,
%number of hidden neurons and momentum
figure;
plot(Grid_MLP(1:5,3),Grid_MLP(1:5,6),'Marker','*')
xlabel('Learning Rate');ylabel('Time');
title('MLP Grid Search: Time v/s Learning Rate')

% saving grid search results. Columns- networkDepth, numHiddenNeurons
%lr, momentum, meanF1ScoreError, totalTime
save('..\Results\Grid_MLP','Grid_MLP');

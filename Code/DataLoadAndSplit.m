close all; clear;clc;

%Loading the data set which has been preprocessed using python
data = table2array(readtable('../Data/Diabetes_preprocessed.csv','ReadVariableNames', true));

input = data(:, 1:end-1);
target = data(:, end);
rng(110);
cv = cvpartition(target,'HoldOut',0.2, 'Stratify',true);
dataTrain = data(cv.training,:);
dataTest  = data(cv.test,:);

%This dataset will only be used for data analysis
save('..\Data\data');

%This dataset will only be used for model training and validation.
save('..\Data\dataTrain');

%This dataset will only be used for final model testing and comparison
save('..\Data\dataTest');

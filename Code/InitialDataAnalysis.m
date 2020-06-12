%Performs Exploratory Data analysis for the dataset

%Clear workspace, command window and close figures
clear all;
clc;
close all;

% Import data
table = readtable('../Data/Diabetes_preprocessed.csv','ReadVariableNames', true);
data = table2array(table);
input = data(:, 1:end-1);
target = data(:, end);

target = categorical(target);

numericColumns = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'};

fprintf('\n**Exploratory Data Analysis on Pima Diabetes Data**\n')

%Determining the dimensions of the dataset
[m n] = size(data);
fprintf('The dataset has %d Rows and %d Columns.\n', m, n)
fprintf('The dataset has %d Numeric and %d Categorical attributes.\n',...
    length(numericColumns),n-length(numericColumns) )

%Summarising the data set min, max, median
x=summary(table)

%Plotting the target class distribution
labels = {'0','1'};
figure(1)
pie(target)
title('Target Class Distribution')

%Histogram
% Univariate Analysis : Histograms of Features
figure('pos',[10 10 1000 600])
title('Histogram of features')
for col_index = 1:n-1
   subplot(4,2,col_index)
   histogram(data(:, col_index))
   title(sprintf('Histogram of %s', numericColumns{col_index}))
end

% Computing the Correlation Matrix (Pearson coefficient)
figure(3)
corrMatrix = corr(input,'type','Pearson');

% Plot the Correlation Matrix
xvalues = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction', 'Age'};
yvalues = {'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'PedigreeFunction', 'Age'};
title('Correlation Chart')
heatmap(xvalues,yvalues,corrMatrix);
title('Correlation Plot (Pearson) - Numeric Attributes')
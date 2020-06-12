%This script performs algorithm comparison between the best models in MLP and SVM on the
%test dataset
clc;close all;clear;

%Loading the test dataset and models
load('..\Data\dataTest');
load('..\Models\mlpTunedModel1_Noise_ADASYN')
load('..\Models\svm_linear')

%Extraction of features and labels
features = dataTest(:, 1:end-1);
labels = dataTest(:, end);
labels_encoded = dummyvar(categorical(labels));

%%
%Evaluating the best MLP model on the test dataset and printing the
%accuracy and F1 score
mlpTunedModel1_Noise_ADASYN_prediction=mlpTunedModel1_Noise_ADASYN(features');
[~,con]=confusion(labels_encoded',mlpTunedModel1_Noise_ADASYN_prediction); 
[accuracyMLP1_ADA_Noise,precisionMLP1, recallMLP1, specificityMLP1,fscoreMLP1_ADA_Noise] = PerformanceMetrics(con);
fprintf('\nMLP1Noise:F1score=%g Accuracy =%g\n)', fscoreMLP1_ADA_Noise,accuracyMLP1_ADA_Noise);

%%
%Evaluating the best SVM model on the test dataset and printing the
%accuracy and F1 score
[svm_linear_prediction,PostProbs] = predict(svm_linear, features);
con=confusionmat(labels',svm_linear_prediction'); 
[accuracySVMLinear,precisionSVMLinear, recallSVMLinear, specificitySVMLinear,fscoreSVMLinear] = PerformanceMetrics(con);
fprintf('\nSVM Linear:F1score=%g Accuracy =%g\n)', fscoreSVMLinear,accuracySVMLinear);
%%
%Plotting the confusion matrix and ROC curve
figure;plotconfusion(labels',svm_linear_prediction','SVM on Test Set',labels_encoded',mlpTunedModel1_Noise_ADASYN_prediction,'MLP on Test Set')
figure;plotroc(labels_encoded',PostProbs','SVM',labels_encoded',mlpTunedModel1_Noise_ADASYN_prediction,'MLP')
%%
%Comparison of F1 score and accuracy metrics between MLP and SVM
figure;vals=[accuracyMLP1_ADA_Noise accuracySVMLinear ; fscoreMLP1_ADA_Noise fscoreSVMLinear];
b= bar(vals);
title('Performance comparison between All Models on Test set')
names = {'Accuracy','F1Score'};
xticklabels(names)
ylim([0,1])
legend('MLP','SVM Linear')
%hold on
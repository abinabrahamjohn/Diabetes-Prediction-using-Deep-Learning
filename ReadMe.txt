Author:Abin Abraham

**Instruction**

To run the model evaluation on the test data set, please run Evaluation.m script. This will run the evaluation of the best 2 models on the test data 'dataTest' which is under 'Data' folder.

Please ensure that the current folder in matlab points to the content of the Code folder location.

**All scripts are organized as folders with the following content**

Data
All data - original, preprocessed, training and test data is available in the ‘Data’ folder. 

Code
All Matlab Code is available in ‘Code’ folder. 

Preprocessing
Preprocessing python script availabe in 'Preprocessing' folder.

Models
Best models are available in ‘Models’ folders. These are the models which were selected as part of the model validation and selection process.

Results
Grid Search and Bayesian optimization results are available in ‘Results’ folder.

Figures
All diagram generated from the scripts


**Details on scripts**
There are multiple MATLAB scripts which are run to arrive at the final models. They can be categorized as below:
1.	Data analysis – InitialDataAnalysis.m

2.	Data Load and Split – DataLoadAndSplit.m

3.	1st level Hyper parameter tuning using Grid Search and Time complexity study:
a.	MLP_GridSearch.m
b.	SVM_Linear_GridSearch.m 
c.	SVM_Gaussian_GridSearch.m
d.	SVM_Poly_GridSearch

4.	2nd level Hyper parameter tuning using Bayesian optimization and training best models on full training data
a.	MLP_Hyper_BayesOpt.m
b.	SVM_Linear_Hyper_BayesOpt.m
c.	SVM_Gaussian_Hyper_BayesOpt.m
d.      SVM_Poly_Hyper_BayesOpt.m

Note that running this script will generate models for all kernels. The currrent model file only has the best model ie linear svm and mlp.The others were discarded.

5.	Evaluation of best models from MLP and SVM - Evaluation.m

6.	Helper functions
a.	ADASYN.m (Source: Author: Dominic Siedhoff , https://uk.mathworks.com/matlabcentral/fileexchange/50541-adasyn-improves-class-balance-extension-of-smote)
b.	PerformanceMetrics.m
c.	[ModelName]KFold _F1_Score Loss.m

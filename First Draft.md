FIRST DRAFT/ NO DATA OR ANALYSIS/ ONLY CONCEPTUAL DOCUMENTATION FRAMEWORK WRITTEN

# Unguided, simply permutational performance comparison between NNs and SVMs
## Biola CSCI480 Machine Learning Professor Buzi
## Subin Kim, April 29th 2020
### Overview
- DATASET  
  The dataset used is keras [MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits)  
  It has been divided into S_out 10000 datapoints and S_in 60000 datapoints which is also subsequently divided into S_val 10000 and S_train 50000.  
`(60000, 28, 28) (10000, 28, 28) (50000, 28, 28) (10000, 28, 28)`  
- GOAL  
  The goal of the project is to train the S_train datapoints using ~~undefined~~ iterations of permutations of NN layers and ~~undefined~~ iterations of permutations of SVM parameters to compare their out of sample error of S_out.
- APPROACH
  1. Neural Networks  
    Keras Sequential models will be used. First layer will consist of a Flatten which expands the 2D pixel data of 28x28 into a 1D 784 neural nodes. ~~Undefined~~ number of Dense layers will be used with varying activation methods ranging from 'relu', 'softmax', 'tanh', 'sigmoid' etc. Through an arbitrary choice of nodes for these dense layers, the optimal combination with the highest accuracy will be chosen.
  2. Support Vector Machines
    Non linear hyperplane kernel 'rbf' will be constantly used. Parameters: gamma and C will be the control variables which will be determining the optimal combination of values. The values of gamma and C will vary logarithmically from 0.01 ~ 1000.
- TREATMENT    
  10 fold cross validation will be applied to all loop iterations of different NN layer combinations and SVM parameter combinations. This process takes the average of each chosen 'validation fold' with the remaining 9 training folds for a total of 10 times each loop to take the mean of the accuracy of prediction. Keep in mind that there exists a meta-layer of validation process since S_train is initially testing against S_val already.
### Results
- NN  
  Chosen model:  
  S_out:  
- SVM  
  Chosen model:  
  S_out:  
### Discussion

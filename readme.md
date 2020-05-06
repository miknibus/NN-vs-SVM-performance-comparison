# Unguided, simply permutational performance comparison between NNs and SVMs
## Biola CSCI480 Machine Learning Professor Buzi
## Subin Kim, April 29th 2020
### Overview
- DATASET  
  The dataset used is keras [MNIST](https://keras.io/datasets/#mnist-database-of-handwritten-digits)  
  It has been divided into S_out 10000 datapoints and S_in 60000 datapoints which is also subsequently divided into S_val 10000 and S_train 50000.  
`(60000, 784) (10000, 784) (50000, 784) (10000, 784)`  
- GOAL  
  The goal of the project is to train the S_train datapoints using 12 iterations of permutations of NN layers and ~~undefined~~ iterations of permutations of SVM parameters to compare their out of sample error of S_out.
- APPROACH
  1. Neural Networks  
    Keras Sequential models will be used. First layer will consist of a Flatten which expands the 2D pixel data of 28x28 into a 1D 784 neural nodes. 2-3 number of Dense layers will be used with varying activation methods ranging from 'relu64', 'relu128', 'selu64', 'selu128', 'tanh64', 'tanh128', 'sigmoid64', 'sigmoid128', 'elu64', 'elu128', 'linear64', 'linear128'. The number behind the activation method indicate the nodes: 64 indicates 2 dense layers, 128 indicate a single dense layer with respective number of nodes.
  2. Support Vector Machines
    Non linear hyperplane kernel 'rbf' will be constantly used. Parameters: gamma and C will be the control variables which will be determining the optimal combination of values. 
     - TREATMENT    
        10 fold cross validation will be applied to all loop iterations of different SVM parameter combinations. This process takes the average of each chosen 'validation fold' with the remaining 9 training folds for a total of 10 times each loop to take the mean of the accuracy of prediction.
### Results
- NN  
    ```
                    loss  accuracy
    relu64      0.114324    0.9653
    relu128     0.104213    0.9710
    selu64      0.136109    0.9578
    selu128     0.124806    0.9639
    tanh64      0.113714    0.9671
    tanh128     0.106669    0.9669
    sigmoid64   0.159434    0.9519
    sigmoid128  0.149648    0.9551
    elu64       0.120642    0.9647
    elu128      0.120707    0.9643
    linear64    0.358552    0.9025
    linear128   0.291578    0.9194
    ```
- SVM  
    Fitting 10 folds for each of 9 candidates, totalling 90 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 8 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed: 206.6min
    [Parallel(n_jobs=-1)]: Done  90 out of  90 | elapsed: 455.2min finished
    GridSearchCV(cv=KFold(n_splits=10, random_state=None, shuffle=False),
                error_score=nan,
                estimator=SVC(C=1.0, break_ties=False, cache_size=200,
                            class_weight=None, coef0=0.0,
                            decision_function_shape='ovr', degree=3,
                            gamma='scale', kernel='rbf', max_iter=-1,
                            probability=False, random_state=None, shrinking=True,
                            tol=0.001, verbose=False),
                iid='deprecated', n_jobs=-1,
                param_grid=[{'C': [5, 10, 15], 'gamma': [0.01, 0.001, 0.0001]}],
                pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
                scoring='accuracy', verbose=1)

    [9 rows x 32 columns]
    The best test score is 0.9803599999999999 corresponding to hyperparameters {'C': 15, 'gamma': 0.01}

### Discussion  
  - NN [Reference](https://www.tensorflow.org/guide/keras/train_and_evaluate)
    - Even though I have no actual idea what those activations mean, (I did read the [documentation](https://keras.io/activations/), but it did not change anything significant about the understanding) I will try my best from a general statistical observation standpoint to give feedback about the results. (I mean as mentioned above in the title, this is "Unguided, simply permutational performance comparison")
    - The biggest observation I can make is that there is a general increase in accuracy and a decrease in loss between the number of layers. The 128 denses have a single layer and 64 denses have a double layer. Which I do not understand at all since my understanding is that the more varied the layers the better at reaching a correct solution because it leaves to diversification to gradually curve out the error deviance. Maybe because the number of nodes actually play a higher factor in determining the performance than the number of layers because the double layers have half=64 the single layer's nodes=128.
    - The only other thing that can be observed is that linear activation is, as expectedly, very poor in performance in compared to the other curved ones. This is pretty obvious because as we learned in class, curved ones mean that they are higher in degrees of freedom, therefore creating more space of coverage in dimension.  
  - SVM [Reference](https://www.kaggle.com/nishan192/mnist-digit-recognition-using-svm)
    - Unlike the Neural Network code, which I understand what almost every single line does, I do not understand what is actually happening for the SVM code with high confidence. So I have added a lot of comment about the code of what I think it is doing and how the output reflects the code. The reason I have chosen this reference code is to reduce downtime where I simply sit around for 4 hours waiting for the process to complete and find out that there was an error, only to fix a couple of lines and wait for another 4 hours. This way it ensured that the code at least functions and I get to test the ranges of hyperparameters without having to waste time debugging and waiting. 
    - Increasing the folds exponentially increased the time it took for the process to complete. Therefore, I started with 2 fold tests to find the correct range of C and gamma.
    - GridsearchCV practically is essentially the ultimate pre-packaged method which searches the combinations of given hyperparameters C and gamma in a grid search with cross validation. This allows us from not using the nested for loops going through each iteration of C and gamma arrays. But to be honest, the grid search is practically doing the same nested looping operation.
    - The pandas method Dataframe was used in the NN too which literally makes a framework of data. Based on the dimensions of the given data, the columns and row indexes can be specified to display the data in pretty much excel format.
    - GridsearchCV already has attributes available for us to access, namely: best_score_ and best_params_. This pretty much automatically records the best accuracy and the C and gamma values corresponding to that score. We can also manually find this information from the dataframe object by searching for the rank_test_score=1.
    - Since I am using multiprocessors, I have added the parameter n_jobs=-1 to utilize all 8 concurrent workers to speed up the process. 
    - To my overjoyed surprise, the multiprocessor concurrency proved to be the solution to this time complexity issue. Before, each run would take multiple hours to complete, or worse not complete at all. So I had reduced the training set size to 10000 just so I could see any sort of results. But after finding out that using 8 concurrent workers brought the time of 10 folds CV with N=10000 down to 20 minutes, I am currently running the full dataset of N=50000 with 10 folds. This creates a matrix of 9 instances with 90 total fits to process. 
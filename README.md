# Metric-BRE

Here we uploaded our source codes and datasets for the project of the course COMS6998 Advanced Data Structures. 

Since the original dataset is too large, we only uploaded the low-dimensional data encoded by a neural network. And with the same reason, we did not upload the codes for actual image search, i.e. you cannot input an image and expect to see some more here. Instead, we uploaded the codes for evaluation. These codes run the entire algorithms, check the results with known labels, and give the statistics of search accuracy. 

There are two kinds of codes here: 

1. The first kind is for parameters optimization in data preprocessing. These codes normally take several hours to get the results in our testing environment on Google Cloud. These codes include: 

MetricLearning.py - Information-theoretic metric learning. Output A.npy
MetricLearningNormalized.py - Information-theoretic metric learning for normalized data. Output A_normal.npy
WOptimization.py - Optimization for Matrix W used in BRE. Output W.npy
WOptimizationMetric.py - Optimization for Matrix W used in Metric-BRE. Output W_metric.npy

2. The second kind is for evaluation. These codes normally run one to a few minutes in our testing environment on Google Cloud. Each one is for one approach and outputs the accuracy statistics. You do not need to run the optimization codes first to run the evaluations. We uploaded all the necessary intermediate data here. These codes include: 

MetricBRE.py - Metric-learning-based BRE method, which is also our main method. Input A_normal.npy and W_metric.npy. Output stat_MetricBRE.txt
Exhaustive.py - Exhaustive method. Output stat_Exhaustive.txt
BRE.py - BRE method. Input W.npy. Output stat_BRE.txt
MetricLSH.py - Metric-learning-based LSH. Input A.npy. Output stat_MetricLSH.txt
MetricRandom.py - Metric-learning-based random hashing. Input A.npy. Output stat_MetricRandom.txt
EuclideanRandom.py - Random hashing based on Euclidean distance. Output stat_EuclideanRandom.txt

All the codes are in Python. To run these codes please make sure your computer already has Python and relevant packages like numpy and scipy installed. 

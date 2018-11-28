# Data Mining Parallel GPU Algorithms

This is the repo for EE 361C Multicore Term Project on Data Mining algorithms using CUDA. The algorithms we chose to implement are:
- Multiple Linear Regression (Closed Form Solution)
- Ridge Regularized Regression (Closed Form Solution)
- Logistic Regression (Iterative)
- KMeans Unsupervised Clustering (Iterative)

Each subdirectory has a readme, associated Makefile, and notebook or C main. 

For KMeans and LogReg, there is an associated jupyter notebook that can be read from GitHub that shows time analysis and graphs, and also the test cases in datasets. For LinearRegression, the test case can be ran from ./obj/linreg, which uses test cases of various size for testing and time complexity analysis. 

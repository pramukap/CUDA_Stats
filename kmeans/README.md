# KMeans CUDA Implementation

This subdirectory is the parallel CUDA implementation of iterative k-means unsupervised learning algorithm. 

### Setup 
This assumes you have NVIDIA graphics card, nvcc, python3, numpy, pandas, matplotlib, jupyter, sklearn, Make

### Running
First run the make file to generate the shared file object kmeans.so, then run the iris_demo notebook, or write your own using the KMeans class from clustering.py

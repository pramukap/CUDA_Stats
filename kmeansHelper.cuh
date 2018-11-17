#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void subtractPointFromMeans(double *points, double *centroids, int m, int n, int k, int point);
__global__ void addPointToMeans(double *points, double *centroids, int m, int n, int k, int point);
__global__ void getDistances(double *vectors, double *distances, int k, int n);
__global__ void assignClass(double *distances, double* labels, int k, int i);
__global__ void findNewCentroids(double *points, double *centroids, double* classlabels, int m, int n, int k, int* counts);
__global__ void divide_by_count(double *centroids, int *count, int n, int k);
__global__ void init_zeros(double *array);

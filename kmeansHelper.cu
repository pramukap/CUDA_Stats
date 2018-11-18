#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// takes a data point (1xn) [x0, x1, x2, x3, x4]
// and subtracts this vector from all means:
//  [k11 - x1, k12 - x2, k13 - x3, k14 - x4]
//  [k21 - x1, k22 - x2, ...
//  [k31 - x1, k32 - x2 ....
//   ...
//  [kk1 - x1, ...

// points = data
// centroids = k means
// m = num observations
// n = num features
// k = num k means
// point = point under scrutiny

// requires centroid to be init
__global__ void subtractPointFromMeans(double *points, double *centroids, int m, int n, int k, int point) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int f = x % k;
    centroids[x] = centroids[x] - points[point * n + f];
}

// adding back the subtraction to get the original k means
__global__ void addPointToMeans(double *points, double *centroids, int m, int n, int k, int point) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int f = x % k;
    centroids[x] = centroids[x] + points[point * n + f];
}

// Called by K threads
__global__ void getDistances(double *vectors, double *distances, int k, int n){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int distance = 0;
    for(int i = 0; i < n; i++){
        distance += vectors[x*n + i];
    }
    distances[x] = distance;
}

__global__ void assignClass(double *distances, int* labels, int k, int i){

    int min = distances[0];
    int minidx = 0;
    for (int i = 0; i < k; i++){
        if (distances[i] < min){
            min = distances[i];
            minidx = i;
        }
    }
    labels[i] = minidx;

}

// counts has to be initialized to zeros
__global__ void findNewCentroids(double *points, double *centroids, int* classlabels, int m, int n, int k, int* counts){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int label = classlabels[x / n];

    atomicAdd(counts+label, 1);
    atomicAdd(centroids+k * label + x % n, points[x]);

}

__global__ void divide_by_count(double *centroids, int *count, int n, int k){

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    centroids[x] = centroids[x] / (count[x / n] * n);

}


__global__ void init_zeros(double *array){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = 0;

}

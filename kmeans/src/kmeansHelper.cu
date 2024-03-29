#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <stdio.h>


__global__ void assignClasses(double *data, double *means, int m, int n, int k, int*labels, double*distances){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int distance = 0;
	for(int ki = 0; ki < k; ki++){
		// For mean ki
		for(int i = 0; i < n; i++){
			distance += (data[idx*n + i]-means[ki*n + i])*(data[idx*n + i]-means[ki*n + i]);
		}
		if (ki == 0 || distance < distances[idx]){
			labels[idx] = ki;
			distances[idx] = distance;
		}
		distance = 0;
	}

}


// requires centroid to be init
__global__ void subtractPointFromMeans(double *points, double *centroids, int m, int n, int k, int point) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int f = x % n;
    centroids[x] = centroids[x] - points[point * n + f];
}

// adding back the subtraction to get the original k means
__global__ void addPointToMeans(double *points, double *centroids, int m, int n, int k, int point) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int f = x % n;
    centroids[x] = centroids[x] + points[point * n + f];
}

// Called by K threads
__global__ void getDistances(double *vectors, double *distances, int k, int n){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int distance = 0;
    for(int i = 0; i < n; i++){
        distance += vectors[x*n + i]*vectors[x*n + i];
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

// PROVIDED BY NVIDIA
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// counts has to be initialized to zeros
__global__ void findNewCentroids(double *points, double *centroids, int* classlabels, int m, int n, int k, int* counts){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int label = classlabels[x / n];

    atomicAdd(counts+label, 1);
    atomicAdd(centroids+ n*label + x % n, points[x]);

}

__global__ void divide_by_count(double *centroids, int *count, int n, int k){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if(count[x/n] == 0){
        //printf("DIVIDE BY ZERO!!!");
    }
    centroids[x] = n * centroids[x] / (count[x / n] );

}


__global__ void init_zeros(double *array){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = 0;

}

__global__ void init_zeros(int *array){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    array[idx] = 0;

}

__global__ void init_labels(int* labels, int k){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    labels[x] = x % k;

}

__global__ void copyCentroidToOld(double* newM, double* old){ 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if( !(newM[idx] != newM[idx]) ){
        old[idx] = newM[idx];
    }

}

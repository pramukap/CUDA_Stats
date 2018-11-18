#include "kmeansHelper.cu"
#include "cuda_runtime.h"
#include <stdio.h>


void kmeans(double* data, int m, int n, int k, double* centroids, int iterations){
    
    
    double *data_d;
    double *centroids_d;
    int *counts;
    int *labels;
    double *distances;
    
    cudaMalloc((void**)&data_d, m*n*sizeof(double));
    cudaMalloc((void**)&centroids_d, k*n*sizeof(double));
    cudaMalloc((void**)&counts, k*sizeof(int));
    cudaMalloc((void**)&labels, m*sizeof(int));
    cudaMalloc((void**)&distances, k*sizeof(double));

    cudaMemcpy(data_d, data, m*n*sizeof(double), cudaMemcpyHostToDevice);

    init_labels<<<n, 1>>>(labels, k);
    // Update Means Step
    init_zeros<<<k, 1>>>(counts);
    findNewCentroids<<<k, n>>>(data_d, centroids_d, labels, m, n, k, counts);
    divide_by_count<<<k, n>>>(centroids_d, counts, n, k);

    // Set number of iterations
    for(int step__ = 0; step__ < iterations; step__++){ 

        // Assignment Step
        for(int i = 0; i < m; i++){

            subtractPointFromMeans<<<k, n>>>(data_d, centroids_d, m, n, k, i);

            getDistances<<<k, 1>>>(centroids_d, distances, k, n);

            assignClass<<<1, 1>>>(distances, labels, k, i);

            addPointToMeans<<<k, n>>>(data_d, centroids_d, m, n, k, i);

        }            
        
        // Update Means Step
        init_zeros<<<k, 1>>>(counts);

        findNewCentroids<<<k, n>>>(data_d, centroids_d, labels, m, n, k, counts);

        divide_by_count<<<k, n>>>(centroids_d, counts, n, k);

    }

}

void run_small_kmeans_test(){


    int m = 12;
    int n = 2;
    int k = 3;
    int iterations = 10;

    double* data = {0, 1, 1, 0, 1, 1, 5, 6, 6, 7, 5, 5, 0, 8, 1, 9, 0, 9};
    double* centroids;

    kmeans(data, m, n, k, centroids, iterations);

}

int main(){
    
   run_small_kmeans_test(); 
   return 0;
}

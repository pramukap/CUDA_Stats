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
    
    init_zero<<<k, n>>>(centroids);

    // Set number of iterations
    for(int step__ = 0; step__ < iterations; step__++){ 

        // Assignment Step
        for(int i = 0; i < m; i++){

            subtractPointFromMeans<<<k, n>>>(data_d, centroids_d, m, n, k, i);

            getDistances<<<k, 1>>>(centroids_d, distances, k, n);

            assignClass(distances, labels, k, i);

            addPointToMeans<<<k, n>>>(data_d, centroids_d, m, n, k, i);

        }            
        
        // Update Means Step
        init_zero<<<k, 1>>>(counts);

        findNewCentroids<<<k, n>>>(data_d, centroids_d, labels, m, n, k, counts);

        divide_by_count<<<k, n>>>(centroids_d, counts, n, k);

    }

}

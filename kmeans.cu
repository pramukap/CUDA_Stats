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

    init_labels<<<m, 1>>>(labels, k);
    
    // Update Means Step
    init_zeros<<<k, 1>>>(counts);
    init_zeros<<<k, n>>>(centroids_d);

    findNewCentroids<<<m, n>>>(data_d, centroids_d, labels, m, n, k, counts);
    
    divide_by_count<<<k, n>>>(centroids_d, counts, n, k);

    cudaMemcpy(centroids, centroids_d, k*n*sizeof(double), cudaMemcpyDeviceToHost);

    // Set number of iterations
    for(int step__ = 0; step__ < iterations; step__++){ 

        // Assignment Step
        for(int point = 0; point < m; point++){

            subtractPointFromMeans<<<k, n>>>(data_d, centroids_d, m, n, k, point);

            getDistances<<<k, 1>>>(centroids_d, distances, k, n);

            assignClass<<<1, 1>>>(distances, labels, k, point);

            addPointToMeans<<<k, n>>>(data_d, centroids_d, m, n, k, point);

        }            
        
        // Update Means Step
        init_zeros<<<k, 1>>>(counts);
        init_zeros<<<k, n>>>(centroids_d);


        findNewCentroids<<<m, n>>>(data_d, centroids_d, labels, m, n, k, counts);

        divide_by_count<<<k, n>>>(centroids_d, counts, n, k);

    }
    cudaMemcpy(centroids, centroids_d, k*n*sizeof(double), cudaMemcpyDeviceToHost);
    for(int i = 0; i < k; i++){
        printf("\nKmean%d:\t", i);
        for(int j = 0; j < n; j++){
            printf("%f\t", centroids[i*n + j]);
        }
    }

}

void run_small_kmeans_test(){


    int m = 12;
    int n = 2;
    int k = 3;
    int iterations = 5;

    double data[m*n] = {0, 1, 1, 0, 1, 1,0,0, 5, 6, 6, 7,7,5 ,5, 5, 0, 8, 1, 9, 0, 9,1,8};
    double* centroids = (double*) malloc(k*n*sizeof(double));

    kmeans(data, m, n, k, centroids, iterations);

}

int main(){
    
   run_small_kmeans_test(); 
   return 0;
}

#include "kmeansHelper.cu"
//#include "University_Data.h"
#include "Iris_Data.h"
//#include "large_cluster.h"
#include "cuda_runtime.h"
#include <stdio.h>

extern "C" void kmeans(double* data, int m, int n, int k, double* centroids, int iterations);

extern "C" {
// Assume data is filled
// Assume centroids is allocated
void kmeans(double* data, int m, int n, int k, double* centroids, int iterations){

    printf("RUNNING KMEANS: (%dx%d) for %d clusters using %d itr\n", m, n, k, iterations);

    double *data_d;
    double *centroids_d;
    int *counts;
    int *labels;
    double *distances;

    double *new_mean;

    cudaMalloc((void**)&data_d, m*n*sizeof(double));
    cudaMalloc((void**)&centroids_d, k*n*sizeof(double));
    cudaMalloc((void**)&new_mean, k*n*sizeof(double));
    cudaMalloc((void**)&counts, k*sizeof(int));
    cudaMalloc((void**)&labels, m*sizeof(int));
    /* old distance
    cudaMalloc((void**)&distances, k*sizeof(double));
    */
    cudaMalloc((void**)&distances, m*sizeof(double));
    cudaMemcpy(data_d, data, m*n*sizeof(double), cudaMemcpyHostToDevice);

/*
    for(int i = 0; i < m*n; i++){
        printf("%f\t", data[i]);
    }
*/

    // Initalize centroids using random partition of data into k groups
    init_labels<<<m, 1>>>(labels, k);
    init_zeros<<<k, 1>>>(counts);
    init_zeros<<<k, n>>>(centroids_d);

    findNewCentroids<<<m, n>>>(data_d, centroids_d, labels, m, n, k, counts);
    divide_by_count<<<k, n>>>(centroids_d, counts, n, k);


    // Set number of iterations
    for(int step__ = 0; step__ < iterations; step__++){
/*
        cudaMemcpy(centroids, centroids_d, k*n*sizeof(double), cudaMemcpyDeviceToHost);
        for(int i = 0; i < k*n; i++){
            printf("%f\t",centroids[i]);
        }
        printf("\n\n");
*/
        // Assignment Step
        init_zeros<<<m, 1>>>(distances);
        assignClasses<<<m, 1>>>(data_d, centroids_d, m, n, k, labels, distances);

/*
        // OLD ASSIGNMENT
        for(int point = 0; point < m; point++){

            subtractPointFromMeans<<<k, n>>>(data_d, centroids_d, m, n, k, point);
            getDistances<<<k, 1>>>(centroids_d, distances, k, n);
            assignClass<<<1, 1>>>(distances, labels, k, point);
            addPointToMeans<<<k, n>>>(data_d, centroids_d, m, n, k, point);

        }
*/        // Update Means Step
        init_zeros<<<k, 1>>>(counts);
        init_zeros<<<k, n>>>(new_mean);
        findNewCentroids<<<m, n>>>(data_d, new_mean, labels, m, n, k, counts);
        divide_by_count<<<k, n>>>(new_mean, counts, n, k);

        copyCentroidToOld<<<k, n>>>(new_mean, centroids_d);

    }
    cudaMemcpy(centroids, centroids_d, k*n*sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(data_d);
    cudaFree(centroids_d);
    cudaFree(counts);
    cudaFree(labels);
    cudaFree(distances);

}

// Assume centroids is filled
// Assume labels is allocated
// Assume data is filled
void kmeans_classify(double * centroids, double * data, int *labels_h, int m, int n, int k){

    double *data_d;
    double *centroids_d;
    int *labels;
    double *distances;
/*
    printf("RUNNING CLASSIFY m%d\tn%d\tk%d\t\n", m, n, k);

    for(int i = 0; i < k*n; i++){
        printf("%f\t",centroids[i]);
    }
    for(int i = 0; i < m*n; i++){
        printf("%f\t",data[i]);
    }
    for(int i = 0; i < m; i++){
        printf("%d\t",labels_h[i]);
    }
*/
    cudaMalloc((void**)&data_d, m*n*sizeof(double));
    cudaMalloc((void**)&centroids_d, k*n*sizeof(double));
    cudaMalloc((void**)&labels, m*sizeof(int));
    cudaMalloc((void**)&distances, k*sizeof(double));

    cudaMemcpy(data_d, data, m*n*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(centroids_d, centroids, k*n*sizeof(double), cudaMemcpyHostToDevice);

    printf("RUNNING CLASSIFY\n");

    // Assignment Step
    init_zeros<<<m, 1>>>(distances);
    assignClasses<<<m, 1>>>(data_d, centroids_d, m, n, k, labels, distances);

    cudaMemcpy(labels_h, labels, m*sizeof(int), cudaMemcpyDeviceToHost);

/*    for(int i = 0; i < m; i++){
        printf("%d\t",labels_h[i]);
    }
*/
}
}
void run_small_kmeans_test(){

    int m = 12;
    int n = 2;
    int k = 3;
    int iterations = 100;

    double data[m*n] = {0, 1, 1, 0, 1, 1,0,0, 5, 6, 6, 7,7,5 ,5, 5, 0, 8, 1, 9, 0, 9,1,8};
    double* centroids = (double*) malloc(k*n*sizeof(double));

    kmeans(data, m, n, k, centroids, iterations);

    for(int i = 0; i < k*n; i++){
        printf("%f\n", centroids[i]);
    }

}

void printConfusionMatrix(int *actual, int*expect){

    int tp = 0;
    int fp = 0;
    int fn = 0;
    int tn = 0;

    for(int i = 0; i < 777; i++){
        if(actual[i] == expect[i]){
            if(actual[i] == 1){
                tp++;
            } else {
                tn++;
            }
        } else {
            if (actual[i] == 1){
                fp++;
            }else{
                fn++;
            }
        }
    }
    printf("\n");
    printf("TP: %d\nFP: %d\nFN: %d\nTN: %d\n", tp, fp, fn, tn);


}
/*
void run_uni_data_test(){

    int m = 777;
    int n = 17;
    int k = 2;
    int iterations = 10;

    double* centroids = (double*) malloc(k*n*sizeof(double));
    kmeans(data, m, n, k, centroids, iterations);
    for(int i = 0; i < k; i++){
        printf("\nKmean%d:\t", i);
        for(int j = 0; j < n; j++){
            printf("%f\t", centroids[i*n + j]);
        }
    }

    int *labels = (int *) malloc(sizeof(int) * m);
    kmeans_classify(centroids, data, labels, m, n, k);

    printConfusionMatrix(labels, results);
}*/

void run_iris_data(int itr_){

    int m = 150;
    int n = 4;
    int k = 3;
    int itr = itr_;

    double *centroids = (double*) malloc(k*n*sizeof(double));
    kmeans(data, m, n, k, centroids, itr);

    for(int i = 0; i < k; i++){
        printf("\nKmean%d:\t", i);
        for(int j = 0; j < n; j++){
            printf("%f\t", centroids[i*n + j]);
        }
    }


}

void run_iris_data_output(int itr_){

    int m = 150;
    int n = 4;
    int k = 3;
    int itr = itr_;

    double *centroids = (double*) malloc(k*n*sizeof(double));
    kmeans(data, m, n, k, centroids, itr);

    for(int i = 0; i < k; i++){
        if(i==2){
        for(int j = 0; j < n; j++){
            if(j != n -1){
            printf("%f,", centroids[i*n + j]);
            } else {
                printf("%f", centroids[i*n + j]);
            }
        }}
    }
    int *labels = (int *) malloc(sizeof(int) * m);
    kmeans_classify(centroids, data, labels, m, n, k);
    for(int i = 0; i < m; i++){
        if (i == m-1){
            printf("%d", labels[i]);
        } else{
        printf("%d,", labels[i]);
        }}


}

void run_large_dataset(int m_){

    int m = m_;
    int n = 2;
    int k = 15;
    int itr = 500;

    clock_t start, end;
    double cpu_time;

    double *centroids = (double*) malloc(k*n*sizeof(double));

    start = clock();
    kmeans(data, m, n, k, centroids, itr);
    end = clock();

    cpu_time = ((double)(end-start))/CLOCKS_PER_SEC;

    printf("Size: %d ran in %f\n", m, cpu_time);
    for(int i = 0; i < k; i++){
        printf("\nKmean%d:\t", i);
        for(int j = 0; j < n; j++){
            printf("%f\t", centroids[i*n + j]);
        }
    }
    
}

int main(){
    run_iris_data_output(20);
    return 0;
}
/*
int main(){
   
    
    run_large_dataset(10);
    run_large_dataset(100);
    run_large_dataset(500);
    run_large_dataset(1000);
    run_large_dataset(2500);
    run_large_dataset(5000);
 //  run_small_kmeans_test();
 //  run_uni_data_test();
//    for(int i = 0; i < 12; i++){
 //       printf("\n");
   //     run_iris_data_output(i);
   // }
   return 0;
}*/ 

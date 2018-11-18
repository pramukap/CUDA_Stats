#include "kmeansHelper.cu"
#include "University_Data.h"
#include "cuda_runtime.h"
#include <stdio.h>

void kmeans(double* data, int m, int n, int k, double* centroids, int iterations);
void kmeans_classify(double * centroids, double * data, int *labels_h, int m, int n, int k);
void run_small_kmeans_test();
void printConfusionMatrix(int *actual, int*expect);
void run_uni_data_test();

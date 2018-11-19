#include "University_Data.h"
#include "time.h"
#include "kmeans.cu"

void run_uni_data_test(int size){

    clock_t start, end;
    double cpu_time_used;

    int m = size;
    int n = 17;
    int k = 2;
    int iterations = 10;

    double* centroids = (double*) malloc(k*n*sizeof(double));
    start = clock();
    kmeans(data, m, n, k, centroids, iterations);
    end = clock();

    cpu_time_used = ((double)(end-start))/CLOCKS_PER_SEC;

    printf("Time used for size %d:\t%f\n", size, cpu_time_used);
}

int main(void){

    run_uni_data_test(10);
    run_uni_data_test(100);
    run_uni_data_test(1000);
    run_uni_data_test(10000);

    return 0;
}

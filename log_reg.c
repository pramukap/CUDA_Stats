#include    "cuda_runtime.h"
#include    "matrixFunctions.cuh"
#include    "vec_kernels.cuh"
#include    "math.h"

#define     BLOCKSIZE = 1024

void    fit(double *X, double *y, double *theta, double lr, size_t n, size_t m, size_t n_iter) 
{
    // Clear the coefficients
    memset(theta, 0, sizeof(double) * m);
    size_t grid_size = (n + BLOCKSIZE - 1) / BLOCKSIZE;

    for (size_t i = 0; i < n_iter; i++) {
        double *z;
        cudaMalloc(&o, sizeof(double) * grid_size);
        vec_dot_product<<<BLOCKSIZE, grid_size>>>(X, theta, o, 1, n);
        cudaDeviceSynchronize();
        double *h;
        cudaMalloc(&h, sizeof(double) * grid_size);
        vec_sigmoid<<<BLOCKSIZE, grid_size>>>(o, h, 1, grid_size);
        cudaDeviceSynchronize();
        double *gradient;
        cudaMalloc(&gradient, sizeof(double) * grid_size);
        
    }
}

#include    "cuda_runtime.h"
#include    "matrixFunctions.cuh"
#include    "vec_kernels.cuh"
#include    "math.h" 

#include    "stddef.h"

#define     BLOCKSIZE       1024
#define     GRIDSIZE(d)     (((d) + ((BLOCKSIZE) - 1)) / (BLOCKSIZE))

void    fit(double *X, double *y, double *theta, double lr, size_t n, size_t m, size_t n_iter) 
{
    double *Xt, *Xd, *yd, *thetad;
    
    cudaMalloc(&Xt, sizeof(double) * n * m);
    
    cudaMalloc(&Xd, sizeof(double) * n * m);
    cudaMemcpy(Xd, X, sizeof(double) * n * m, cudaMemcpyHostToDevice);
    
    cudaMalloc(&yd, sizeof(double) * m);
    cudaMemcpy(yd, y, sizeof(double) * m, cudaMemcpyHostToDevice);
    
    cudaMalloc(&thetad, sizeof(double) * n);
    cudaMemcpy(thetad, theta, sizeof(double) * n, cudaMemcpyHostToDevice);
    MatrixTranspose<<<GRIDSIZE(m*n), BLOCKSIZE>>>(Xd, Xt, m, n);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < n_iter; i++) {
        double *z, *h, *g;
        cudaMalloc(&z, sizeof(double) * n);
        cudaMalloc(&h, sizeof(double) * n);
        cudaMalloc(&g, sizeof(double) * m);

        MatrixMul<<<GRIDSIZE(n), BLOCKSIZE>>>(X, theta, z, n, m, 1, m);
        cudaDeviceSynchronize();

        vec_sigmoid<<<GRIDSIZE(n), BLOCKSIZE>>>(z, h, 1, m);
        cudaDeviceSynchronize();

        vec_scalar_mul<<<GRIDSIZE(n), BLOCKSIZE>>>(h, h, -1.0, 1, m);
        cudaDeviceSynchronize();

        vec_add<<<GRIDSIZE(n), BLOCKSIZE>>>(h, y, h, 1, m);
        cudaDeviceSynchronize();

        MatrixMul<<<GRIDSIZE(n*m), BLOCKSIZE>>>(Xt, h, gradient, n, m, 1, m);
        cudaDeviceSynchronize();

        vec_scalar_mul<<<GRIDSIZE(n), BLOCKSIZE>>>(g, g, -lr / m, 1, m);
        cudaDeviceSynchronize();

        vec_add<<<GRIDSIZE(n), BLOCKSIZE>>>(theta, g, theta, 1, m);
        cudaDeviceSynchronize();

        cudaFree(z);
        cudaFree(h);
        cudaFree(g);
    }

    cudaFree(Xd);
    cudaFree(Xt);
    cudaFree(yd);
    cudaMemcpy(theta, thetad, sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaFree(thetad);
}

double*    predict_proba(double *X, double *theta, size_t m, size_t n)
{
    double *y = (double*) malloc(sizeof(double) * m);
    double *yd;
    double *Xd;
    double *thetad;
    cudaMalloc(&yd, sizeof(double) * m);
    cudaMalloc(&Xd, sizeof(double) * m * n);
    cudaMalloc(&thetad, sizeof(double) * n);
    cudaMemcpy((void*) Xd, (void*) X, sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) thetad, (void*) theta, sizeof(double) * n, cudaMemcpyHostToDevice);

    MatrixMul<<<GRIDSIZE(m * n), BLOCKSIZE>>>(Xd, thetad, yd, n, m, 1, n);
    cudaDeviceSynchronize();

    cudaMemcpy((void*) y, yd, sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaFree(Xd);
    cudaFree(thetad);
    cudaFree(yd);
    return y;
}

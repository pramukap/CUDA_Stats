#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"
#include	"matrixFunctions.cuh"
#include    "math.h" 

#include    <cstddef>
#include	<cstdlib>
#include    <iostream>

#define     BLOCKSIZE       1024
#define     GRIDSIZE(d)     (((d) + ((BLOCKSIZE) - 1)) / (BLOCKSIZE))

__global__
void    mat_transpose(double *X, double *Xt, size_t m, size_t n) 
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= m*n)
        return;
    
    size_t row = gid / n;
    size_t col = gid % n;
    Xt[col * m + row] = X[row * n + col];
}

extern "C"
{

void    fit(double *X, double *y, double *theta, double lr, size_t m, size_t n, size_t n_iter) 
{
    double *Xt, *Xd, *yd, *thetad;
    
    cudaMalloc(&Xt, sizeof(double) * n * m);
    
    cudaMalloc(&Xd, sizeof(double) * n * m);
    cudaMemcpy(Xd, X, sizeof(double) * n * m, cudaMemcpyHostToDevice);
    
    cudaMalloc(&yd, sizeof(double) * m);
    cudaMemcpy(yd, y, sizeof(double) * m, cudaMemcpyHostToDevice);
    
    cudaMalloc(&thetad, sizeof(double) * n);
    cudaMemcpy(thetad, theta, sizeof(double) * n, cudaMemcpyHostToDevice);
    
    mat_transpose<<<GRIDSIZE(m*n), BLOCKSIZE>>>(X, Xt, m, n);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < n_iter; i++) {
        double *z, *h, *g;
        cudaMallocManaged(&z, sizeof(double) * m);
        cudaMallocManaged(&h, sizeof(double) * m);
        cudaMallocManaged(&g, sizeof(double) * n);

        // dot(X, theta)
        vec_dot_mat<<<GRIDSIZE(m), BLOCKSIZE>>>(X, theta, z, m, n);
        cudaDeviceSynchronize();

        // h = sigm(z)
        vec_sigmoid<<<GRIDSIZE(m), BLOCKSIZE>>>(z, h, 1, m);
        cudaDeviceSynchronize();

        // h = -h
        vec_scalar_mul<<<GRIDSIZE(m), BLOCKSIZE>>>(h, h, -1.0, 1, m);
        cudaDeviceSynchronize();

        // h = y - h
        vec_add<<<GRIDSIZE(m), BLOCKSIZE>>>(h, y, h, 1, m);
        cudaDeviceSynchronize();

        // h = -(y - h) = h - y
        vec_scalar_mul<<<GRIDSIZE(m), BLOCKSIZE>>>(h, h, -1.0, 1, m); 

        // g = dot(Xt, h)
        vec_dot_mat<<<GRIDSIZE(n), BLOCKSIZE>>>(Xt, h, g, n, m);
        cudaDeviceSynchronize();

        // g = -(g*lr) / m
        vec_scalar_mul<<<GRIDSIZE(n), BLOCKSIZE>>>(g, g, -(lr / m), 1, n);
        cudaDeviceSynchronize();

        // theta = theta + (-g) = theta - g
        vec_add<<<GRIDSIZE(n), BLOCKSIZE>>>(theta, g, theta, 1, n);
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

void    predict_proba(double *X, double *theta, double *y, size_t m, size_t n)
{
    double *yd;
    double *Xd;
    double *thetad;
    cudaMalloc(&yd, sizeof(double) * m);
    cudaMalloc(&Xd, sizeof(double) * m * n);
    cudaMalloc(&thetad, sizeof(double) * n);
    cudaMemcpy((void*) Xd, (void*) X, sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) thetad, (void*) theta, sizeof(double) * n, cudaMemcpyHostToDevice);

    MatrixMul<<<m, n>>>(Xd, thetad, yd, n, m, 1, n);
    cudaDeviceSynchronize();

    cudaMemcpy((void*) y, yd, sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaFree(Xd);
    cudaFree(thetad);
    cudaFree(yd);
    return y;
}

}

int	main(void)
{
	double *X; // = (double*) malloc(sizeof(double) * 1024 * 1024);
	double *y; // = (double*) malloc(sizeof(double) * 1024);
	double *theta; // = (double*) malloc(sizeof(double) * 1024);

	cudaMallocManaged(&X, sizeof(double) * 1024 * 1024);
	cudaMallocManaged(&y, sizeof(double) * 1024);
	cudaMallocManaged(&theta, sizeof(double) * 1024);
	for (int i = 0; i < 1024; i++) {
		y[i] = i % 2;
		theta[i] = 0;
		for (int j = 0; j < 1024; j++) {
			X[i * 1024 + j] = j;
		}
	}

	fit(X, y, theta, 0.01, 1024, 1024, 1);
}



#include    "cuda_runtime.h"
#include    "vec_kernels.cuh"
#include    "math.h" 
#include	"data.h"

#include    <cstddef>
#include	<cstdlib>
#include    <cassert>
#include    <iostream>
#include    <ctime>

#define     BLOCKSIZE       1024
#define     GRIDSIZE(d)     (((d) + ((BLOCKSIZE) - 1)) / (BLOCKSIZE))

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

    double *z, *h, *g;
	cudaMalloc(&z, sizeof(double) * m);
	cudaMalloc(&h, sizeof(double) * m);
	cudaMalloc(&g, sizeof(double) * n);
    
    mat_transpose<<<GRIDSIZE(m*n), BLOCKSIZE>>>(Xd, Xt, m, n);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < n_iter; i++) {

        // dot(X, theta)
        vec_dot_mat<<<GRIDSIZE(m), BLOCKSIZE>>>(Xd, thetad, z, m, n);
        cudaDeviceSynchronize();

        // h = sigm(z)
        vec_sigmoid<<<GRIDSIZE(m), BLOCKSIZE>>>(z, h, 1, m);
        cudaDeviceSynchronize();

        // h = -h
        vec_scalar_mul<<<GRIDSIZE(m), BLOCKSIZE>>>(h, h, -1.0, 1, m);
        cudaDeviceSynchronize();

        // h = y - h
        vec_add<<<GRIDSIZE(m), BLOCKSIZE>>>(h, yd, h, 1, m);
        cudaDeviceSynchronize();

        // h = -(y - h) = h - y
        vec_scalar_mul<<<GRIDSIZE(m), BLOCKSIZE>>>(h, h, -1.0, 1, m); 
        cudaDeviceSynchronize();

        // g = dot(Xt, h)
        vec_dot_mat<<<GRIDSIZE(n), BLOCKSIZE>>>(Xt, h, g, n, m);
        cudaDeviceSynchronize();

        // g = -(g*lr) / m
        vec_scalar_mul<<<GRIDSIZE(n), BLOCKSIZE>>>(g, g, -(lr / m), 1, n);
        cudaDeviceSynchronize();

        // theta = theta + (-g) = theta - g
        vec_add<<<GRIDSIZE(n), BLOCKSIZE>>>(thetad, g, thetad, 1, n);
        cudaDeviceSynchronize();

    }

    cudaFree(z);
	cudaFree(h);
	cudaFree(g);
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
    cudaMallocManaged(&yd, sizeof(double) * m);
    cudaMalloc(&Xd, sizeof(double) * m * n);
    cudaMalloc(&thetad, sizeof(double) * n);
    cudaMemcpy((void*) Xd, (void*) X, sizeof(double) * m * n, cudaMemcpyHostToDevice);
    cudaMemcpy((void*) thetad, (void*) theta, sizeof(double) * n, cudaMemcpyHostToDevice);

    //MatrixMul<<<m, n>>>(Xd, thetad, yd, n, m, 1, n);
    vec_dot_mat<<<GRIDSIZE(m), BLOCKSIZE>>>(Xd, thetad, yd, m, n);
    cudaDeviceSynchronize();

    vec_sigmoid<<<GRIDSIZE(m), BLOCKSIZE>>>(yd, yd, 1, m);
    cudaDeviceSynchronize();

    cudaMemcpy((void*) y, yd, sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaFree(Xd);
    cudaFree(thetad);
    cudaFree(yd);
}

}

int	main(void)
{
    // Testing with the house_data dataset. label is y >= y.mean()
	int m = 21613, n = 8;
	double *X = (double*) malloc(sizeof(double) * m * n);
	double *y = (double*) malloc(sizeof(double) * m);
	double *theta = (double*) malloc(sizeof(double) * n);

    // Copy into memory
	memcpy(X, X_house, sizeof(double) * 21613 * 8);
	memcpy(y, y_house, sizeof(double) * 21613);

    // Call host function for fit
    fit(X, y, theta, 0.01, m, n, 100);
    
    printf("Theta after 100 iterations: ");
    for (int i = 0; i < n; i++) {
        printf("%f, ", theta[i]);
    }
    printf("\n");

    printf("TEST: Asserting coeffs against known-good values. ");
    for (int i = 0; i < n; i++) {
        assert(abs(theta[i] - known_theta[i]) < 0.01);
    }
    printf("PASSED.\n");

	double *yt = (double*) malloc(sizeof(double) * 21613);
    predict_proba(X, theta, yt, 21613, 8);

    int miss = 0;
    for (int i = 0; i < m; i++) {
        if (abs(yt[i] - known_yt[i]) > 0.01)
            miss++;
    }
    printf("TEST: %d of %d labels differ from known-good logit.\n", miss, m);
    
    printf("TEST: Scaling m (observations) (CSV):\n\nm,cpu_time\n");
    size_t local_m = 21;
    for (int i = 0; i < 4; i++) {
        clock_t start = clock();
        fit(X, y, theta, 0.01, local_m, n, 100);
        clock_t end = clock();
        printf("%d,%f\n", local_m, ((double) (end - start)) / CLOCKS_PER_SEC);
        local_m *= 10;
    }

    free(X);
    free(y);
    free(theta);
    free(yt);
}



#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <time.h>
#include <stdio.h>

//Ax is the number of rows
//Ay is the number of cols
__global__ void meanOfFeatures(double *A, double *means, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	means[x] = 0;	
	
	int i;
	for (i = 0; i < Ax; i++) {
		means[x] += A[x + (i*Ay)];
	}

	means[x] /= Ax;
}

__global__ void subMeansFromCols(double *A, double *B, double *means, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int col = x % Ay;

	B[x] = A[x] - means[col];	
}

//threads allotted is equal to size of covariance matrix
//which is Ay x Ay
__global__ void calcCovMatrix(double *A, double *cov, int Ax, int Ay) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	//feature 1 and feature 2 to correlate
	//corresponds to row and col in covariance matrix
	int f1 = x / Ay;
	int f2 = x % Ay;

	int i;
	for (i = 0; i < Ax; i++) {
		cov[x] += (A[f1 + (i * Ay)] * A[f2 + (i * Ay)]);
	}

	cov[x] /= (Ax - 1);
}	

//https://en.wikipedia.org/wiki/Principal_component_analysis#Covariance-free_computation
//random vector that is of length num_features
__global__ void genRandVector(double *r, curandState *state, unsigned long *clock_seed) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	unsigned long seed = 132241684309342543 + *clock_seed;
	curand_init (seed, x, 0, &state[x]); 
	
	curandState localState = state[x];
	r[x] = (double)curand_uniform(&localState);
	state[x] = localState;
}

__global__ void findEigenVectors(int iterations) { 
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	for (i = 0; i < iterations; i++) {
		
	}
}

#define SIZE 5

int main() {
	double r[5] = {0, 0, 0, 0, 0};		
	unsigned long clock_seed = (unsigned long)clock();

	double *r_d;
	curandState *state_d;
	unsigned long *clock_seed_d;

	cudaMalloc((void**)&r_d, SIZE * sizeof(double));
	cudaMalloc((void**)&state_d, SIZE * sizeof(curandState));
	cudaMalloc((void**)&clock_seed_d, sizeof(unsigned long));

	cudaMemcpy(r_d, r, SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(clock_seed_d, &clock_seed, sizeof(unsigned long), cudaMemcpyHostToDevice);

	genRandVector<<<5, 1>>>(r_d, state_d, clock_seed_d);

    cudaMemcpy(r, r_d, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

	int i;
	for (i = 0; i < SIZE; i++) {
		printf("%f ", r[i]);
	}
	printf("\n");
}

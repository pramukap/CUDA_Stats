#include "cuda_runtime.h"
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
__global__ void genRandVector(double *r) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	r[x] = 
}

__global__ void findEigenVectors( 
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	int i;
	for (i = 0; i < iterations; i++) {
		
	}
